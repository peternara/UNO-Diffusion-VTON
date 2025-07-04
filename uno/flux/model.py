# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .modules.layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock, timestep_embedding


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params       = params
        self.in_channels  = params.in_channels
        self.out_channels = self.in_channels
        
        if params.hidden_size % params.num_heads != 0: # hidden_size가 num_heads로 나누어떨어지는지 체크
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
            
        pe_dim = params.hidden_size // params.num_heads # 1-head 당 embedding 차원
        if sum(params.axes_dim) != pe_dim:              # positional embedding 차원 검증
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
            
        self.hidden_size = params.hidden_size                                                # 전체 임베딩 차원
        self.num_heads   = params.num_heads                                                  # multi-head attention 헤드 개수
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim) # N차원 positional embedding 생성기
        self.img_in      = nn.Linear(self.in_channels, self.hidden_size, bias=True)          # 이미지 입력을 hidden_size로 임베딩 (Linear layer)
        self.time_in     = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)              # 타임스텝 입력(timestep/noise level)을 임베딩 (MLP)
        self.vector_in   = MLPEmbedder(params.vec_in_dim, self.hidden_size)                  # 벡터 입력(예: CLIP 등)을 임베딩 (MLP)
        self.guidance_in = (MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()) # guidance 인풋이 있을 때 MLP로 임베딩, 없으면 그대로 통과
        self.txt_in      = nn.Linear(params.context_in_dim, self.hidden_size)                # 텍스트 임베딩 입력 (예: T5 등) → hidden_size로 투영

        # DoubleStreamBlock(Transformer 계열 블록) 여러 개를 쌓음 (깊이 = params.depth)
        #     → 보통 이미지 스트림/텍스트 스트림 등 두 개 정보를 동시에 처리
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )
        
        # SingleStreamBlock 블록 여러 개 
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )
        
        # 마지막 출력 레이어 (hidden → out_channels)
        self.final_layer            = LastLayer(self.hidden_size, 1, self.out_channels)

        # gradient checkpointing(메모리 절약용 연산) 기능 사용 여부 (기본 False)
        self.gradient_checkpointing = False 

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self):
        # set recursively
        processors = {}  # type: dict[str, nn.Module]

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        ref_img: Tensor | None = None, 
        ref_img_ids: Tensor | None = None, 
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)

        # concat ref_img/img
        img_end = img.shape[1]
        if ref_img is not None:
            if isinstance(ref_img, tuple) or isinstance(ref_img, list):
                img_in = [img] + [self.img_in(ref) for ref in ref_img]
                img_ids = [ids] + [ref_ids for ref_ids in ref_img_ids]
                img = torch.cat(img_in, dim=1)  
                ids = torch.cat(img_ids, dim=1)
            else:
                img = torch.cat((img, self.img_in(ref_img)), dim=1)  
                ids = torch.cat((ids, ref_img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing:
                img, txt = torch.utils.checkpoint.checkpoint(
                    block,
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe, 
                    use_reentrant=False,
                )
            else:
                img, txt = block(
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe
                )

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            if self.training and self.gradient_checkpointing:
                img = torch.utils.checkpoint.checkpoint(
                    block,
                    img, vec=vec, pe=pe,
                    use_reentrant=False
                )
            else:
                img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]
        # index img
        img = img[:, :img_end, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
