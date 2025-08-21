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
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ref_img: None | Tensor=None,
    pe: Literal['d', 'h', 'w', 'o'] ='d'
) -> dict[str, Tensor]:
    assert pe in ['d', 'h', 'w', 'o']
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if ref_img is not None:
        _, _, ref_h, ref_w = ref_img.shape
        ref_img = rearrange(ref_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if ref_img.shape[0] == 1 and bs > 1:
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)
        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)
        # img id分别在宽高偏移各自最大值
        h_offset = h // 2 if pe in {'d', 'h'} else 0
        w_offset = w // 2 if pe in {'d', 'w'} else 0
        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None] + h_offset
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :] + w_offset
        ref_img_ids = repeat(ref_img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    if ref_img is not None:
        return {
            "img": img,
            "img_ids": img_ids.to(img.device),
            "ref_img": ref_img,
            "ref_img_ids": ref_img_ids.to(img.device),
            "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
            "vec": vec.to(img.device),
        }
    else:
        return {
            "img": img,
            "img_ids": img_ids.to(img.device),
            "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
            "vec": vec.to(img.device),
        }

# ----------------------------------------------------------------------------------------
# 메인 이미지와 참조 이미지들(ref_imgs)을 patch 단위로 분해
# 각 이미지 patch에 고유한 positional embedding ID(img_ids, ref_img_ids) 생성
# 텍스트 프롬프트를 두 임베더(t5, clip)로 임베딩
# 이 모든 정보를 dict로 묶어 downstream 모델 입력으로 준비
# ----------------------------------------------------------------------------------------
def prepare_multi_ip(
    t5: HFEmbedder,                       # 텍스트 임베딩용, 예) T5 모델
    clip: HFEmbedder,                     # 텍스트 임베딩용, 예) CLIP 모델
    img: Tensor,                          # 입력 이미지 (B, C, H, W)
    prompt: str | list[str],              # 텍스트 프롬프트 (str 혹은 str 리스트)
    ref_imgs: list[Tensor] | None = None, # 참조 이미지들 (각각 (B, C, H, W)), 기본값 None
    pe: Literal['d', 'h', 'w', 'o'] = 'd' # positional embedding ID 할당 방식 (default 'd')
) -> dict[str, Tensor]:
    #
    assert pe in ['d', 'h', 'w', 'o']
    bs, c, h, w = img.shape
    
    # 만약 이미지 배치가 1인데 prompt가 리스트(여러 개)라면, 배치 사이즈를 prompt 길이로 설정 (프롬프트 수만큼 이미지 복제 예상)
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    # 1. 입력 이미지 Patchify 처리 & ID 생성    
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) # 이미지를 2x2 patch 단위로 분해 → shape: (B, H//2 * W//2, C*2*2)
    if img.shape[0] == 1 and bs > 1: # 이미지 배치가 1이고 실제 배치가 여러개면
        # 만약 배치가 1이고, bs > 1이면 이미지를 bs개로 복제
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    # 메인 이미지 patch별 positional embedding ID 생성

    # [
    #    [[0,0,0], [0,0,0]],
    #    [[0,0,0], [0,0,0]]
    # ]
    img_ids         = torch.zeros(h // 2, w // 2, 3)                  # 패치별 positional embedding id 초기화 (H/2, W/2, 3)

    # row, col에 각 patch의 좌표를 기록
    # 아래 두 코드 실행의 의미
    #     img_ids[..., 0] : 아직 비어 있음 (보통 다른 정보 넣으려고 비워둔 자리)
    #     img_ids[..., 1] : row index (세로 위치)
    #     img_ids[..., 2] : col index (가로 위치)
    #
    # 아래 소스에서, 
    #    > torch.arange(h // 2)[:, None] # shape (2,1)
    #        > = [[0],
    #             [1]]
    #    > img_ids[..., 1] 에 더하면, broadcasting 에 의해, 2번째 열에(row) [0], [1]이 들어감
    # [
    #    [[0,0,0], [0,0,0]],
    #    [[0,1,0], [0,1,0]]
    # ]
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None] # 두 번째 축에 row 인덱스 할당

    # 아래 소스에서, 
    #    > torch.arange(w // 2)[None, :]  # shape (1,2)
    #       > = [[0,1]] 
    #    > → broadcasting으로 col 인덱스가 [...,2]에 들어감 
    # [
    #    [[0,0,0], [0,0,1]],
    #    [[0,1,0], [0,1,1]]
    # ]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :] # 세 번째 축에 col 인덱스 할당

    #
    img_ids         = repeat(img_ids, "h w c -> b (h w) c", b=bs)     # shape을 (B, H/2*W/2(패치수), 3)으로 복제

    # 2. 참조 이미지 처리 및 ID 생성
    ref_img_ids            = []              # 참조 이미지의 positional id 저장 리스트
    ref_imgs_list          = []              # 패치화된 참조 이미지 저장 리스트
    pe_shift_w, pe_shift_h = w // 2, h // 2  # 참조 이미지 id offset (겹침 방지)
    #
    for ref_img in ref_imgs:                 # 참조 이미지 순회
        _, _, ref_h1, ref_w1 = ref_img.shape # 각 참조 이미지 shape 추출
        ref_img = rearrange(ref_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) # 2x2 패치 분해
        if ref_img.shape[0] == 1 and bs > 1:                                               # 참조 이미지도 배치가 1인데 여러 개 필요하면
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)                            # 복제
        ref_img_ids1 = torch.zeros(ref_h1 // 2, ref_w1 // 2, 3)                            # 참조 이미지 패치 id 초기화
        
        # img id 각각 row, col에 offset 부여 (겹치지 않게 위치 이동)
        h_offset             = pe_shift_h if pe in {'d', 'h'} else 0                       # 높이 offset
        w_offset             = pe_shift_w if pe in {'d', 'w'} else 0                       # 참조 이미지 패치 id 초기화
        #
        ref_img_ids1[..., 1] = ref_img_ids1[..., 1] + torch.arange(ref_h1 // 2)[:, None] + h_offset  # row id + offset
        ref_img_ids1[..., 2] = ref_img_ids1[..., 2] + torch.arange(ref_w1 // 2)[None, :] + w_offset  # col id + offset
        ref_img_ids1         = repeat(ref_img_ids1, "h w c -> b (h w) c", b=bs)                      # 배치로 복제
        ref_img_ids.append(ref_img_ids1)
        ref_imgs_list.append(ref_img)

        # pe shift - 다음 참조 이미지는 위치 offset을 누적 (겹치지 않게) >> offset 누적 (다음 참조이미지와 id 겹치지 않게)
        pe_shift_h += ref_h1 // 2
        pe_shift_w += ref_w1 // 2

    if isinstance(prompt, str):                      # 프롬프트가 문자열이면 리스트로 변환
        prompt = [prompt]        
    txt = t5(prompt)                                 # t5 임베더로 텍스트 임베딩    
    if txt.shape[0] == 1 and bs > 1:                 # 임베딩 배치가 1이고 실제 bs > 1이면 > 복제
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)  #     → 복제
    txt_ids = torch.zeros(bs, txt.shape[1], 3)       # 텍스트 토큰별 positional id (dummy)

    vec = clip(prompt)                               # clip 임베더로 텍스트 임베딩
    if vec.shape[0] == 1 and bs > 1:                 # 배치가 1이고 실제 bs > 1이면 > 복제
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,                                                               # (B, 패치수, 패치채널)
        "img_ids": img_ids.to(img.device),                                        # 이미지 patch별 positional id
        "ref_img": tuple(ref_imgs_list),                                          # 참조 이미지 패치화 결과
        "ref_img_ids": [ref_img_id.to(img.device) for ref_img_id in ref_img_ids], # 참조 이미지 patch별 id
        "txt": txt.to(img.device),                                                # t5 임베딩 결과 
        "txt_ids": txt_ids.to(img.device),                                        # 텍스트 토큰별 positional id
        "vec": vec.to(img.device),                                                # clip 임베딩 결과
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    ref_img: Tensor=None,
    ref_img_ids: Tensor=None,
):
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            ref_img=ref_img,
            ref_img_ids=ref_img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec
        )
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
