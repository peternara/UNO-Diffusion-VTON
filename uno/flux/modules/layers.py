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

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from ..math import attention, rope
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------
# 패치나 토큰의 위치(좌표) 정보를, 신경망에서 쓸 수 있는 벡터(임베딩)로 바꿔주는 역할
#     → 그중에서도 이미지 패치처럼 2차원/3차원 위치 정보를 ‘RoPE’ 방식(고급 sinusoidal)으로 각각 임베딩하고,
#     → 이것을 하나로 합쳐줍니다.
# 위치를 벡터로 전환
# 예)
#     → 이미지 패치의 좌표를 임베딩할 때, 즉, 한 이미지 패치의 위치: (row=3, col=7)이라면,
#     → row좌표 3, col좌표 7을 받아, row용 임베딩, col용 임베딩을 각각 만듦 (Sin/Cos) < RoPE 
#     → 두 벡터를 연결해서 최종 positional embedding 
# ---------------------------------------------------------------------------------------

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim      = dim       # head당 positional embedding 차원
        self.theta    = theta     # RoPE의 주기 조절 파라미터
        self.axes_dim = axes_dim  # 각 축별 positional embedding 차원

    def forward(self, ids: Tensor) -> Tensor:
        # ids: (..., n_axes)  ← n_axes는 보통 2 또는 3 (ex: [row, col], 또는 [dummy, row, col] 등)

        n_axes = ids.shape[-1]

        #
        # rope(ids[..., i], self.axes_dim[i], self.theta): (..., axes_dim[i])  
        #     → n_axes번 for문: 각 축별로 positional embedding 생성 (예: row/col/dummy 각각)
        #     → 결과: [ (..., axes_dim[0]), (..., axes_dim[1]), ... ]  
        # torch.cat(dim=-3): 마지막에서 3번째 축(= 시퀀스/패치 방향)에서 concat  
        # (실제 (..., axes_dim[0]+axes_dim[1]+...)로 연결)
        emb    = torch.cat(
                [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
                dim=-3,
                )
    
        return emb.unsqueeze(1)

# ---------------------------------------------------------------------------------------
# Diffusion/Transformer에서 사용되는 sinusoidal positional encoding (위치 임베딩)
#     → 각 배치별 timestep(노이즈 단계)을 고유한 벡터로 변환
#     → t가 0이면 첫 위치, t가 크면 점점 높은 주파수로 포지셔널 정보 제공
#     → 이 임베딩은 네트워크에 timestep 정보를 넣을 때, 학습 가능한 임베딩 레이어 대신 사용
# 각 배치별 timestep(예: 노이즈 단계)을, sin/cos으로 만든 고유 벡터(임베딩)로 바꿔주는 함수!
# ---------------------------------------------------------------------------------------
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings. 
    #
    Sinusoidal timestep embedding 생성
    :param t: (N,) 배치별 timestep 인덱스(실수 가능)
    :param dim: 출력 임베딩 차원
    :param max_period: 임베딩 주파수 범위 제어
    :return: (N, dim) 형태의 임베딩 텐서 반환
    """
    # (N,) - timestep 값에 스케일 적용, 
    t     = time_factor * t  # (N,)  ← 각 배치별 timestep에 time_factor 곱함
    
    # 임베딩의 절반 차원 (cos, sin 각각 사용) 
    half  = dim // 2         # 절반 차원 (예: dim=256이면 half=128)
    
    # sin/cos 에 들어가는 angle 구하기 = 각 배치별 t * 각 주파수
    #
    # (half,) - 각 주파수별 계수(exp scale factor) 
    #     → 각 주파수 : 각 차원의 주파수 값이 다름
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
    # (N, half) - 각 배치별 t * 각 주파수
    #     → t[:, None]: (N, 1)
    #     → freqs[None]: (1, half)
    args  = t[:, None].float() * freqs[None] # (N, half) ← 각 배치의 t와 각 차원의 freq 곱

    # (N, dim) - [cos 부분, sin 부분]으로 이어붙여 최종 임베딩 생성
    #     → torch.cos(args): (N, half)
    #     → torch.sin(args): (N, half)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # concat 결과: (N, half*2) = (N, dim)  ← sin/cos 합치면 원하는 임베딩 차원 완성
    
    if dim % 2:
        # dim이 홀수면 마지막에 0 패딩 (shape 맞추기, (N, dim))
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1) # (N, dim+1) → 마지막 1개 0패딩 (dim이 홀수일 때만) > 결과적으로 (N, dim) 보장
        
    if torch.is_floating_point(t):
        # t가 float 타입이면 embedding도 동일 타입으로 변환
        embedding = embedding.to(t)
    return embedding # (N, dim)  ← 최종 결과, 배치별 timestep 임베딩


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return ((x * rrms) * self.scale.float()).to(dtype=x_dtype)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x

class LoraFluxAttnProcessor(nn.Module):

    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight


    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x) + self.qkv_lora(x) * self.lora_weight
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x) + self.proj_lora(x) * self.lora_weight
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    def forward():
        pass

#---------------------------------------------------------------------------------------------------------------------------------
# adaLN-Zero Block
# 역할: → 세 가지 modulation 파라미터를 담는 단순 데이터 구조.    
#     → shift : 피처를 얼마나 이동(translation)
#     → scale : 피처의 크기를 얼마나 조절
#     → gate  : 피처에 얼마나 "gate"를 걸지 (곱해서 정보를 유입/차단)
#
# what? adaLN-Zero Block
#     → 보통 "FiLM" 레이어와 유사하게, Condition(벡터)에 따라 feature를 affine transform하기 위함
#         → 이 연구에서 scale값은 앞에 나온 vector에 곱해주는 역할이고, shift는 더해주는 역할
#         → 여기서, FiLM에서 나온 linear modulation 방법을 LN (layer normalization)에 적용한 형태가 바로 adaLN이라고 부르는 것
#     → DiT에서 사용되는 adaLN의 경우에는 직접적으로 learnable하는 것이 아닌, timestep과 label의 embedding을 shift와 scale값으로 활용
#     → adaLN은 각 2개의 shift와 scale factor가 필요했다. 즉 총 4개의 embedding vector가 MLP로 출력.
#     → 그러나 adaLN-Zero는 scale factor a를 추가하여서 총 6개의 output이 나오도록 모델 구조를 설계.
#     → 또한, 이 scale factor a의 초기값을 zero로 두고 시작하기 때문에, adaLN-Zero라고 부름.
#         → a가 0이기 때문에 input_tokens 값만 살아남게 되므로, 논문에서 언급하는 것처럼, 처음 DiT block은 identity function.
#         → DiT block 안에 있는 MLP들은 SiLU와 linear layer를 적용하는데, (아래, class Modulation(nn.Module) 참조)
#             → adaLN 또는 adaLN-Zero인지에 따라서 output 차원이 달라진다. 
#             → adaLN-Zero일 경우에는 transformer's hidden size의 6배에 해당하는 vector를 출력하게 된다.
#         → Timesteps과 layer 정보에 대하여 embed로 들어오면 서로 dim-256 사이즈의 vector인데, 두 개의 vector를 더한 상태로 MLP의 입력에 해당.
#---------------------------------------------------------------------------------------------------------------------------------
@dataclass
class ModulationOut:
    # modulation의 shift, scale, gate는
    # (batch, 1, dim)
    #     → 여기서 "1"은 sequence length가 아니라 broadcast를 위한 차원

    # shift
    #     → feature의 "위치"를 이동 (translation)
    #     → LayerNorm, Residual 등에서 bias 역할
    # 예시:
    #     → x_mod = x + shift
    shift: Tensor # (batch, 1, dim)
    
    # scale
    #      → feature의 "크기"를 조절 (amplitude)
    #      → 특정 조건에서 feature를 더 강조하거나, 축소할 수 있음
    # 예시:
    #      → x_mod = (1 + scale) * x
    #          → (1 + scale)를 쓰는 이유: scale=0이면 "원래 피처 유지", scale>0이면 강조, <0이면 약화
    scale: Tensor # (batch, 1, dim)

    # gate
    # 정보의 흐름 자체를 조절 (0~1, sigmoid/softplus 등)
    # attention, MLP 등 여러 가지 결과값을 residual에 더할 때
    # "이 조건에서 이 정보를 얼마나 반영할지"를 제어
    # gate=1이면 다 반영, 0이면 차단
    # 예시:
    #       → x = x + gate * (attention_output)
    #       → StyleGAN의 style strength, diffusion의 classifier-free guidance 등도 gate와 유사한 메커니즘
    gate: Tensor  # (batch, 1, dim)

#-----------------------------------------------------------------------------------
# Condition 정보(vec)에 따라 "feature에 affine 변환(shift, scale)"
# 정보의 양을 gate로 조절
#     → 다양한 residual/MLP/attention 등 여러 블록에서 feature를 “조건에 맞게” 실시간으로 조정할 수 있도록 함
#-----------------------------------------------------------------------------------
class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double  = double
        self.multiplier = 6 if double else 3                               # double이면 shift/scale/gate 2세트, 아니면 1세트 → 곱할 개수 결정
        #
        # lin: condition 벡터를 shift/scale/gate 여러 개로 projection
        # 예) double=True이면 Linear(512, 3072) (== 512 x 6)
        self.lin        = nn.Linear(dim, self.multiplier * dim, bias=True) # 입력: (batch, dim) → 출력: (batch, multiplier*dim)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        #
        # 1. SiLU 활성화: (batch, dim)
        # 2. Linear 투사: (batch, multiplier*dim)
        # 3. unsqueeze(1): (batch, 1, multiplier*dim)
        # 4. chunk: multiplier개로 분리 → 각 (batch, 1, dim) → 여기서 "1"은 sequence length가 아니라 broadcast를 위한 차원
        # 분해 설명
        #    → nn.functional.silu(vec): (batch, dim) → (batch, dim)
        #    → self.lin(...): (batch, dim) → (batch, multiplier*dim)
        #        → multiplier=6 or 3
        # [:, None, :]:
        #    → (batch, multiplier*dim) → (batch, 1, multiplier*dim)
        #    → broadcasting 용 (나중에 feature와 더할 때 시퀀스 차원과 맞추기 위해)
        # .chunk(self.multiplier, dim=-1)
        #    → (batch, 1, multiplier*dim) → multiplier개의 (batch, 1, dim) 텐서로 분해
        #    → 예) [shift1, scale1, gate1, shift2, scale2, gate2]
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class DoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * (attn.img_attn.proj(img_attn) + self.proj_lora1(img_attn) * self.lora_weight)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * (attn.txt_attn.proj(txt_attn) + self.proj_lora2(txt_attn) * self.lora_weight)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, **attention_kwargs):
        # "이 vec를 modulation에 넣는 이유?" and "왜 Modulation을 해야 할까?"
        #    → 이전 앞 부분에서 vec = vec + self.vector_in(y)
        #        → vec : clip embedding
        #        → y   : timestep embedding
        #        → 추가적으로 class guidance 를 들어 갈수 있음 
        #            → vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        #    → “지금 어떤 조건(예: timestep=200, ‘cat’이라는 텍스트 등) 하에서 피처를 어떻게 바꿔야 하나?”를 모델이 바로 인식하도록 해야함.
        #        → 예전 방식 : 그냥 조건을 concat하거나 add만 해줌
        #    → Modulation
        #        → 각 조건에 따라 피처의 “스케일, 위치, 정보 유입량”을 미세하게 바꿀 수 있음
        #        → 즉, 모델이 condition을 "진짜 강하게" 인식할 수 있음
        #        → 대표적인 예시: FiLM (Feature-wise Linear Modulation) 레이어
        img_mod1, img_mod2  = attn.img_mod(vec)
        txt_mod1, txt_mod2  = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated       = attn.img_norm1(img)
        img_modulated       = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv             = attn.img_attn.qkv(img_modulated) # qkv() 한 번에 projection → chunk로 Q, K, V 분리
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k        = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated       = attn.txt_norm1(txt)
        txt_modulated       = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv             = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k        = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

# ----------------------------------------------------------------------------------------------------------------
# "Scalable Diffusion Models with Transformers" 의 MM-DIT 모델 기반
# 주의) 그래서, 기존 SD의 구조와는 다름. 
#    → FLUX 라이브러리는 CLIP을 condition으로 쓸 때, cross-attention을 사용하지 않고, 직접적인 modulation(injection) 구조
#    → 실제 소스에서도 확인 → text, image concat 후, self-attention에 적용함
# ----------------------------------------------------------------------------------------------------------------
class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim   = int(hidden_size * mlp_ratio)
        self.num_heads   = num_heads
        self.hidden_size = hidden_size
        self.head_dim    = hidden_size // num_heads

        self.img_mod     = Modulation(hidden_size, double=True)
        self.img_norm1   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn    = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp     = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
            )

        self.txt_mod     = Modulation(hidden_size, double=True)
        self.txt_norm1   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn    = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp     = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
            )
        
        processor         = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
    ) -> tuple[Tensor, Tensor]:
        if image_proj is None:
            return self.processor(self, img, txt, vec, pe)
        else:
            return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)


class SingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = output + self.proj_lora(torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weight
        output = x + mod.gate * output
        return output


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, **attention_kwargs) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output

class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, x, vec, pe)
        else:
            return self.processor(self, x, vec, pe, image_proj, ip_scale)



class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
