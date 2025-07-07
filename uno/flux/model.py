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
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim) # N차원 positional embedding (patch 위치, 토큰 위치 등)
        self.img_in      = nn.Linear(self.in_channels, self.hidden_size, bias=True)          # 이미지 입력을 hidden_size로 임베딩 (Linear layer)
        self.time_in     = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)              # 타임스텝 입력(timestep/noise level)을 임베딩 (MLP)
        self.vector_in   = MLPEmbedder(params.vec_in_dim, self.hidden_size)                  # 벡터 입력(예: CLIP 등)을 임베딩 (MLP)
        self.guidance_in = (MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()) # guidance 인풋이 있을 때 MLP로 임베딩, 없으면 그대로 통과
        self.txt_in      = nn.Linear(params.context_in_dim, self.hidden_size)                # 텍스트 임베딩 입력 (예: T5 등) → hidden_size로 투영

        # DoubleStreamBlock(Transformer 계열 블록) 여러 개를 쌓음 (깊이 = params.depth)
        #     → 보통 이미지 스트림/텍스트 스트림 등 두 개 정보를 동시에 처리
        self.double_blocks = nn.ModuleList(
            [
                # 트랜스포머 계열 블록으로, 이미지/텍스트 스트림을 동시에 cross-attention 등으로 처리
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
                # concat된 시퀀스를 한 번에 처리하는 추가 트랜스포머 블록
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
        img: Tensor,                        # (B, N_patch, in_channels)   : 입력 이미지 패치 벡터
        img_ids: Tensor,                    # (B, N_patch, pe_dim)        : 이미지 patch별 positional id
        txt: Tensor,                        # (B, N_token, context_in_dim): 텍스트 임베딩
        txt_ids: Tensor,                    # (B, N_token, pe_dim)        : 텍스트 토큰 positional id
        timesteps: Tensor,                  # (B,)                        : diffusion timestep 값
        y: Tensor,                          # (B, vec_in_dim)             : 텍스트 임베딩 등 condition 벡터
        guidance: Tensor | None = None,     # (B,)                        : guidance strength 값
        ref_img: Tensor | None = None,      # (B, N_patch, in_channels)   : 참조 이미지
        ref_img_ids: Tensor | None = None,  # (B, N_patch, pe_dim)        : 참조 이미지 positional id
    ) -> Tensor:
        
        # 입력 이미지, 텍스트 shape 체크
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        # 이미지 입력: Linear(in_channels → hidden_size)
        img = self.img_in(img) # (B, N_img_patch, hidden_size)    

        # * timestep embedding > 이해가 잘 안감?!
        # 타임스텝(예: diffusion step) 임베딩 + 선형 레이어 통과
        # i) 타임스텝 임베딩은 모델이 "지금 어느 단계에 있는지"를 알 수 있도록 신호를 주는 과정.
        #        → 이 신호가 없으면, 모델은 어떤 단계에서 어떤 처리를 해야 하는지 몰라서 제대로 학습/추론 안됨 
        #        → 1000 step diffusion에서 ① 초반(노이즈 많음) ② 후반(노이즈 적음) 단계별로 모델이 "지금 어느 단계인지를 알아야" 제대로 이미지를 복원/생성
        #        → 단순히 timestep=55 같은 정수 값을 모델에 주면, 딥러닝 네트워크가 이 숫자의 상대적인 위치/관계를 잘 이해하지 못함.
        #        → 그래서 Embedding(예: sin, cos 기반 포지셔널 임베딩 또는 Learnable Embedding)으로 **숫자를 "의미 있는 벡터"**로 바꿔줍니다.
        # ii) timestep(scalar) → 256차원 임베딩 → hidden_size로 투영
        #        → timestep_embedding(timesteps, 256) : 각 배치별 timestep(예: 노이즈 단계)을, sin/cos으로 만든 고유 벡터(임베딩)로 바꿔주는 함수!
        #            → timesteps (ex. 10, 50, 900 ...)를 256차원 벡터로 변환 (sin/cos 또는 learnable 등)
        #            → 이 벡터는 각 타임스텝별로 독특한 "위치 인식 신호"를 가짐
        # iii) positional embedding과 비슷한 이유
        #        → timestep embedding과 positional embedding은 “정수 인덱스(위치)”를 “벡터”로 변환한다는 점에서 거의 동일한 원리로 작동
        #            → positional embedding 
        #                → Transformer 등에서 “시퀀스 내 각 토큰의 위치” (예: 0, 1, 2, 3...) 를 벡터로 변환
        #                → 모델이 "순서" 정보를 이해할 수 있도록 해줌
        #            → timestep embedding:
        #                → Diffusion 등에서 “현재 단계(스텝)” (예: 0, 1, ..., 999) 를 벡터로 변환
        #                → 모델이 "지금 어느 단계인지" 알 수 있도록 해줌
        #            → 둘 다
        #                → “정수(인덱스, 단계)” → “분산된 벡터 신호”로 변환
        #                → 주로 Sinusoidal 함수(sin, cos)나 학습 가능한 Embedding Layer 사용
        vec = self.time_in(timestep_embedding(timesteps, 256)) # timestep_embedding(timesteps, 256): (B, 256) → time_in: (B, hidden_size)
        
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            # guidance 플래그가 있으면 guidance값도 임베딩해서 vec에 더함
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256)) # timestep_embedding(guidance, 256): (B, 256) → guidance_in: (B, hidden_size)

        # CLIP 등 condition 벡터도 임베딩해서 더함 
        #        → 왜 여기서 vec를 더 하는 이유는? 
        #            → 보통 SD의 condition은 cross-attention할대 이런식으로 더하지 않음
        #            → 예측)
        #                → condiation에 들어가는 것이 맞나? 아닐수도 있을거 같다.
        vec = vec + self.vector_in(y) # vector_in(y): (B, hidden_size), broadcasting sum 
        
        # 텍스트 임베딩: Linear(context_in_dim → hidden_size)    
        txt = self.txt_in(txt)        # (B, N_txt, hidden_size)         

        # 텍스트 토큰 id와 이미지 패치 id를 시퀀스 차원에서 이어붙임
        #        → paper 수식 4)
        ids = torch.cat((txt_ids, img_ids), dim=1) # txt_ids: (B, N_txt, pe_dim), img_ids: (B, N_img_patch, pe_dim) → (B, N_txt + N_img_patch, pe_dim)

        # concat ref_img/img
        img_end = img.shape[1] # 원본 img 패치 길이 저장
        if ref_img is not None:
            if isinstance(ref_img, tuple) or isinstance(ref_img, list):
                # i) 여러 참조 이미지가 있을 때: 모두 임베딩 후 원본 img와 함께 concat
                #        → paper 수식 5)
                # img: (B, N_img_patch, hidden_size), 
                # self.img_in(ref): (B, N_ref_patch_i, hidden_size)
                img_in  = [img] + [self.img_in(ref) for ref in ref_img]
                
                # ids: (B, N_txt + N_img_patch, pe_dim)
                # ref_ids: (B, N_ref_patch_i, pe_dim)
                img_ids = [ids] + [ref_ids for ref_ids in ref_img_ids]

                # (B, N_txt + N_img_patch + ΣN_ref_patch, hidden_size)
                img     = torch.cat(img_in, dim=1)  

                # (B, N_txt + N_img_patch + ΣN_ref_patch, pe_dim)
                ids     = torch.cat(img_ids, dim=1)
            else:
                # ii) 참조 이미지 1개만 있을 때: img, ids에 이어붙임
                img     = torch.cat((img, self.img_in(ref_img)), dim=1) # img: (B, N_img_patch, hidden_size), self.img_in(ref_img): (B, N_ref_patch, hidden_size) 
                                                                        #         → (B, N_img_patch + N_ref_patch, hidden_size)
                ids     = torch.cat((ids, ref_img_ids), dim=1)          # ids: (B, N_txt + N_img_patch, pe_dim), ref_img_ids: (B, N_ref_patch, pe_dim)
                                                                        #         → (B, N_txt + N_img_patch + N_ref_patch, pe_dim)                
        # RoPE : positional embedding (입력된 id 기준)   
        pe = self.pe_embedder(ids) # (B, N_total_seq, hidden_size // num_heads), 내부적으로 expand됨
        
        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing: # 메모리 절약을 위한 gradient checkpointing
                img, txt = torch.utils.checkpoint.checkpoint(
                    block,
                    img=img,             # (B, N_total_img, hidden_size)
                    txt=txt,             # (B, N_txt, hidden_size) 
                    vec=vec,             # (B, hidden_size) 
                    pe=pe,               # (B, N_total_seq, pe_dim)  ※ 블록 내부적으로 맞게 reshape
                    use_reentrant=False,
                )
            else: # DoubleStreamBlock 통과 (이미지/텍스트 동시 처리 트랜스포머) 
                img, txt = block(
                    img=img,             # (B, N_total_img, hidden_size)
                    txt=txt,             # (B, N_txt, hidden_size)
                    vec=vec,             # (B, hidden_size) 
                    pe=pe                # (B, N_total_seq, pe_dim) 
                )

        # txt와 img 시퀀스를 이어붙임 (텍스트→이미지)
        img = torch.cat((txt, img), 1) # txt: (B, N_txt, hidden_size), img: (B, N_total_img, hidden_size)
                                       #             → (B, N_txt + N_total_img, hidden_size)
        
        # SingleStreamBlock들 차례대로 통과 (1-stream 트랜스포머 계층)
        for block in self.single_blocks:
            if self.training and self.gradient_checkpointing:
                img = torch.utils.checkpoint.checkpoint(
                    block,
                    img,               # (B, N_txt + N_total_img, hidden_size)
                    vec=vec,           # (B, hidden_size)
                    pe=pe,             # (B, N_total_seq, pe_dim)
                    use_reentrant=False
                )
            else:
                img = block(img, vec=vec, pe=pe) # (B, N_txt + N_total_img, hidden_size)

        # 텍스트 시퀀스 부분 제거, 이미지 부분만 남김
        img = img[:, txt.shape[1] :, ...] # (B, N_total_img, hidden_size) - 텍스트 시퀀스 제외

        # 참조 이미지 concat 후에도, 원본 img 패치 길이만큼 자름
        # index img
        img = img[:, :img_end, ...]       # (B, N_img_patch, hidden_size) - 오리지널 img 패치 부분만 선택

        # 마지막 출력 레이어 통과 (hidden → patch별 예측값)
        img = self.final_layer(img, vec)  # (B, N_img_patch, out_channels*patch_size**2) - 예측값(이미지/노이즈)
        return img # 예측 결과 반환          # (B, N_img_patch, out_channels*patch_size**2)         
