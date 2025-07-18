# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import gc
import itertools
import logging
import os
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from uno.dataset.uno import FluxPairedDatasetV2
from uno.flux.sampling import denoise, get_noise, get_schedule, prepare_multi_ip, unpack
from uno.flux.util import load_ae, load_clip, load_flow_model, load_t5, set_lora

if TYPE_CHECKING:
    from uno.flux.model import Flux
    from uno.flux.modules.autoencoder import AutoEncoder
    from uno.flux.modules.conditioner import HFEmbedder

logger = get_logger(__name__)

def get_models(name: str, device, offload: bool=False):
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def inference(
    batch: dict,
    model: "Flux", t5: "HFEmbedder", clip: "HFEmbedder", ae: "AutoEncoder",
    accelerator: Accelerator,
    seed: int = 0,
    pe: Literal["d", "h", "w", "o"] = "d"
) -> Image.Image:
    ref_imgs = batch["ref_imgs"]
    prompt = batch["txt"]
    neg_prompt = ''
    width, height = 512, 512
    num_steps = 25
    x = get_noise(
        1, height, width,
        device=accelerator.device,
        dtype=torch.bfloat16,
        seed=seed + accelerator.process_index
    )
    timesteps = get_schedule(
        num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=True,
    )
    with torch.no_grad():
        ref_imgs = [
            ae.encode(ref_img_.to(accelerator.device, torch.float32)).to(torch.bfloat16)
            for ref_img_ in ref_imgs
        ]
        inp_cond = prepare_multi_ip(
            t5=t5, clip=clip, img=x, prompt=prompt,
            ref_imgs=ref_imgs,
            pe=pe
        )

        x = denoise(
            model,
            **inp_cond,
            timesteps=timesteps,
            guidance=4,
        )

        x = unpack(x.float(), height, width)
        x = ae.decode(x)

    x1 = x.clamp(-1, 1)
    x1 = rearrange(x1[-1], "c h w -> h w c")
    output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())

    return output_img


def resume_from_checkpoint(
    resume_from_checkpoint: str | None | Literal["latest"],
    project_dir: str,
    accelerator: Accelerator,
    dit: "Flux",
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    dit_ema_dict: dict | None = None,
) -> tuple["Flux", torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, dict | None, int]:
    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint is None:
        return dit, optimizer, lr_scheduler, dit_ema_dict, 0

    if resume_from_checkpoint == "latest":
        # Get the most recent checkpoint
        dirs = os.listdir(project_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        if len(dirs) == 0:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            return dit, optimizer, lr_scheduler, dit_ema_dict, 0
        path = dirs[-1]
    else:
        path = os.path.basename(resume_from_checkpoint)
        

    accelerator.print(f"Resuming from checkpoint {path}")
    lora_state = load_file(os.path.join(project_dir, path, 'dit_lora.safetensors'), device=accelerator.device)
    unwarp_dit = accelerator.unwrap_model(dit)
    unwarp_dit.load_state_dict(lora_state, strict=False)
    if dit_ema_dict is not None:
        dit_ema_dict = load_file(
            os.path.join(project_dir, path, 'dit_lora_ema.safetensors'),
            device=accelerator.device
        )
        if dit is not unwarp_dit:
            dit_ema_dict = {f"module.{k}": v for k, v in dit_ema_dict.items() if k in unwarp_dit.state_dict()}

    global_step = int(path.split("-")[1])
    
    return dit, optimizer, lr_scheduler, dit_ema_dict, global_step

@dataclasses.dataclass
class TrainArgs:
    ## accelerator
    project_dir: str | None = None
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: int = 1
    seed: int = 42
    wandb_project_name: str | None = None
    wandb_run_name: str | None = None

    ## model
    model_name: Literal["flux-dev", "flux-schnell"] = "flux-dev"
    lora_rank: int = 512
    double_blocks_indices: list[int] | None = dataclasses.field(
        default=None,
        metadata={"help": "Indices of double blocks to apply LoRA. None means all double blocks."}
    )
    single_blocks_indices: list[int] | None = dataclasses.field(
        default=None,
        metadata={"help": "Indices of double blocks to apply LoRA. None means all single blocks."}
    )
    pe: Literal["d", "h", "w", "o"] = "d"
    gradient_checkpoint: bool = True
    ema: bool = False
    ema_interval: int = 1
    ema_decay: float = 0.99


    ## optimizer
    learning_rate: float = 1e-2
    adam_betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    ## lr_scheduler
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 100
    max_train_steps: int = 100000

    ## dataloader
    train_data_json: str = "datasets/dreambench_singleip.json" # TODO: change to your own dataset, or use one data syenthsize pipeline comming in the future. stay tuned 
    batch_size: int = 1
    text_dropout: float = 0.1
    resolution: int = 512
    resolution_ref: int | None = None

    eval_data_json: str = "datasets/dreambench_singleip.json"
    eval_batch_size: int = 1

    ## misc
    resume_from_checkpoint: str | None | Literal["latest"] = None
    checkpointing_steps: int = 1000

def main(
    args: TrainArgs,
):
    ## accelerator
    deepspeed_plugins = {
        "dit": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero2_config.json'),
        "t5": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero3_config.json'),
        "clip": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero3_config.json')
    }
    accelerator = Accelerator(
        project_dir=args.project_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        deepspeed_plugins=deepspeed_plugins,
        log_with="wandb",
    )
    set_seed(args.seed, device_specific=True)
    accelerator.init_trackers(
        project_name=args.wandb_project_name,
        config=args.__dict__,
        init_kwargs={
            "wandb": {
                "name": args.wandb_run_name,
                "dir": accelerator.project_dir,
            },
        },
    )
    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }.get(accelerator.mixed_precision, torch.float32)

    ## logger
    logging.basicConfig(
        format=f"[RANK {accelerator.process_index}] " + "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True
    )
    logger.info(accelerator.state)
    logger.info("Training script launched", main_process_only=False)

    ## model
    dit, vae, t5, clip = get_models(
        name=args.model_name,
        device=accelerator.device,
    )
    
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)

    dit.requires_grad_(False)
    dit = set_lora(dit, args.lora_rank, args.double_blocks_indices, args.single_blocks_indices, accelerator.device)
    dit.train()
    dit.gradient_checkpointing = args.gradient_checkpoint

    ## optimizer and lr scheduler
    optimizer = torch.optim.AdamW(
        [p for p in dit.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=args.adam_betas,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_eps,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # dataloader
    dataset = FluxPairedDatasetV2(
        json_file=args.train_data_json,
        resolution=args.resolution, resolution_ref=args.resolution_ref
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    eval_dataset = FluxPairedDatasetV2(
        json_file=args.eval_data_json,
        resolution=args.resolution, resolution_ref=args.resolution_ref
    )   
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn
    )

    dataloader = accelerator.prepare_data_loader(dataloader)
    eval_dataloader = accelerator.prepare_data_loader(eval_dataloader)
    dataloader = itertools.cycle(dataloader)  # as infinite fetch data loader

    ## parallel
    accelerator.state.select_deepspeed_plugin("dit")
    dit, optimizer, lr_scheduler = accelerator.prepare(dit, optimizer, lr_scheduler) 
    accelerator.state.select_deepspeed_plugin("t5")
    t5 = accelerator.prepare(t5)  # type: torch.nn.Module
    accelerator.state.select_deepspeed_plugin("clip")
    clip = accelerator.prepare(clip)  # type: torch.nn.Module
    
    ## ema
    dit_ema_dict = {
        k: deepcopy(v).requires_grad_(False) for k, v in dit.named_parameters() if v.requires_grad
    } if args.ema else None

    ## resume
    (
        dit,
        optimizer,
        lr_scheduler,
        dit_ema_dict,
        global_step
    ) = resume_from_checkpoint(
        args.resume_from_checkpoint,
        project_dir=args.project_dir,
        accelerator=accelerator,
        dit=dit,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dit_ema_dict=dit_ema_dict
    )

    ## noise scheduler
    timesteps = get_schedule(
        999,
        (args.resolution // 8) * (args.resolution // 8) // 4,
        shift=True,
    )
    timesteps = torch.tensor(timesteps, device=accelerator.device)
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total validation prompts = {len(eval_dataloader)}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    train_loss = 0.0
    while global_step < (args.max_train_steps):
        batch = next(dataloader)
        prompts = [txt_ if random.random() > args.text_dropout else "" for txt_ in batch["txt"]]
        img = batch["img"]
        ref_imgs = batch["ref_imgs"]


        #--------------------------------------------------------------------------------------------
        # 이 코드를 이해하기 위해, 다음 연구 기반임을 인식
        #        → 즉, Flux는 "Scalable Diffusion Models with Transformers" 의 MM-DIT 논문 기반
        #            → 이 paper의 Fig.3을 이해야함.
        #            → 기존 SD 버전의 구조, 특히, cross-attention 적용하는 구조가 아님!!
        #--------------------------------------------------------------------------------------------
        with torch.no_grad():
            x_1   = vae.encode(img.to(accelerator.device).to(torch.float32))
            x_ref = [vae.encode(ref_img.to(accelerator.device).to(torch.float32)) for ref_img in ref_imgs]

            # 이미지 Patchify & IDs
            #    → 메인 이미지와 참조 이미지들(ref_imgs)을 patch 단위로 분해
            #    → 각 이미지 patch에 고유한 positional embedding ID(img_ids, ref_img_ids) 생성
            #    → 텍스트 프롬프트를 두 임베더(t5, clip)로 임베딩
            #    → 이 모든 정보를 dict로 묶어 downstream 모델 입력으로 준비
            inp   = prepare_multi_ip(t5=t5, clip=clip, img=x_1, prompt=prompts, ref_imgs=tuple(x_ref), pe=args.pe)
            
            # 입력 이미지 (B, C, H, W) > (B, H//2 * W//2, C*2*2)
            #    → 이미지를 2x2 패치 단위로 분해하여, 각 패치를 하나의 벡터(길이 C*4)
            #    → 결과적으로 (배치, 패치 개수, 패치 채널)  
            # 의문) 아래 부분은 "prepare_multi_ip()" 함수안에서 동일한 패치화를 하는데 왜 여기서 또 하는가?
            #    → 실제, 아래 x_1와 동일한 inp['img']을 사용하지 않음.
            x_1   = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)      
            # 참조 이미지 리스트(각각 (B, C, H, W) 형태)의 각 이미지를 위와 동일하게 2x2 패치 단위로 분해
            #    → 리스트 내 모든 참조 이미지를 (B, H//2 * W//2, C*2*2) 분해
            x_ref = [rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for x in x_ref]

            # What?
            # 각 배치에 대해 무작위 타임스텝을 정해서,
            # 해당 타임스텝만큼 "노이즈"가 섞인 이미지를 x_t로 생성.
            # (이 때 원본 이미지 x_1과 노이즈 x_0을 혼합함)
            # diffusion 네트워크의 입력이 되는 x_t와, 각종 conditioning 정보를 준비
            bs           = img.shape[0]

            # Add noise at image
            # (0~999) 사이에서 배치별로 무작위 타임스텝(timestep) 인덱스를 생성.
            #    → 즉, 각 이미지가 'diffusion 과정 중 몇 번째 단계'에 해당할지 랜덤하게 결정.
            #    → Diffusion 모델에서는 다양한 타임스텝을 골고루 학습시키기 위해 랜덤 추출.
            t            = torch.randint(0, 1000, (bs,), device=accelerator.device)
            # 위에서 뽑은 인덱스를 실제 timestep 값(timesteps 텐서에서 해당 값)으로 변환.
            #    → 보통 timesteps는 [0, 1] 구간의 값들을 쭉 담고 있는 텐서.
            #    → 예: linspace(0, 1, 1000)와 동일
            #    → 즉, t는 각 배치마다 0.0~1.0 사이의 연속적 노이즈 비율 값으로 생성
            t            = timesteps[t]
            # x_1과 같은 크기(shape)의 랜덤 노이즈 이미지를 생성. (완전 랜덤 노이즈) > backword pass > 노이즈 제거 과정
            #    → Diffusion 모델의 핵심은 노이즈를 추가하거나 제거하는 것이므로, 노이즈 샘플이 필요.
            x_0          = torch.randn_like(x_1, device=accelerator.device)
            # x_1(원본)과 x_0(노이즈)를 타임스텝 비율만큼 섞어서 x_t를 생성하는 역할.
            #    → t=0이면, x_t는 x_1(원본 이미지)
            #    → t=1이면, x_t는 x_0(완전 노이즈)
            #    → 그 사이 값이면 점진적으로 노이즈가 섞인 이미지
            #    → 즉, diffusion training에서 '특정 단계의 noisy image'를 시뮬레이션.
            #    → 이 x_t를 네트워크에 입력해서, 원본(x_1)이나 노이즈(x_0)를 예측하게 만듭니다.
            x_t          = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0
            
            # guidance_vec는 모두 1로 채워진 벡터(배치 크기만큼).
            #    → diffusion 모델에서 '조건부 샘플링'(예: classifier-free guidance) 할 때
            #        → "1"이면 conditioning을 적용하라는 신호.
            #        → (이 값이 0이면 unconditional, 1이면 conditional)
            guidance_vec = torch.full((x_t.shape[0],), 1, device=x_t.device, dtype=x_t.dtype)
        
        with accelerator.accumulate(dit):
            # Predict the noise residual and compute loss
            model_pred = dit(
                img=x_t.to(weight_dtype),                                                       # 노이즈가 섞인 이미지 (배치, 패치, 채널) - weight_dtype(예: float16/bfloat16)로 타입 변환
                img_ids=inp['img_ids'].to(weight_dtype),                                        # 각 이미지 패치의 positional embedding id, dtype 변환                                     
                ref_img=[x.to(weight_dtype) for x in x_ref],                                    # 참조 이미지들(2x2 패치화된 것), dtype 변환
                ref_img_ids=[ref_img_id.to(weight_dtype) for ref_img_id in inp['ref_img_ids']], # 참조 이미지 patch별 id, dtype 변환
                txt=inp['txt'].to(weight_dtype),                                                # 텍스트 임베딩 (T5 등), dtype 변환
                txt_ids=inp['txt_ids'].to(weight_dtype),                                        # 텍스트 토큰별 positional id, dtype 변환
                y=inp['vec'].to(weight_dtype),                                                  # 텍스트 임베딩 (CLIP 등), dtype 변환
                timesteps=t.to(weight_dtype),                                                   # diffusion 타임스텝 값, dtype 변환
                guidance=guidance_vec.to(weight_dtype)                                          # guidance 벡터, dtype 변환           
            )

            loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0

        if accelerator.sync_gradients and dit_ema_dict is not None and global_step % args.ema_interval == 0:
            src_dict = dit.state_dict()
            for tgt_name in dit_ema_dict:
                dit_ema_dict[tgt_name].data.lerp_(src_dict[tgt_name].to(dit_ema_dict[tgt_name]), 1 - args.ema_decay)

        if accelerator.sync_gradients and accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
            logger.info(f"saving checkpoint in {global_step=}")
            save_path = os.path.join(args.project_dir, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)

            # save
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(dit)
            unwrapped_model_state = unwrapped_model.state_dict()
            unwrapped_model_state = {k: v for k, v in unwrapped_model_state.items() if v.requires_grad}

            accelerator.save(
                unwrapped_model_state,
                os.path.join(save_path, 'dit_lora.safetensors'),
                safe_serialization=True
            )
            unwrapped_opt = accelerator.unwrap_model(optimizer)
            accelerator.save(unwrapped_opt.state_dict(), os.path.join(save_path, 'optimizer.bin'))
            logger.info(f"Saved state to {save_path}")

            if args.ema:
                accelerator.save(
                    {k.split("module.")[-1]: v for k, v in dit_ema_dict.items()},
                    os.path.join(save_path, 'dit_lora_ema.safetensors')
                )

            # validate
            dit.eval()
            torch.set_grad_enabled(False)
            for i, batch in enumerate(eval_dataloader):
                result = inference(batch, dit, t5, clip, vae, accelerator, seed=0)
                accelerator.log({f"eval_gen_{i}": result}, step=global_step)


            if args.ema:
                original_state_dict = dit.state_dict()
                dit.load_state_dict(dit_ema_dict, strict=False)
                for batch in eval_dataloader:
                    result = inference(batch, dit, t5, clip, vae, accelerator, seed=0)
                    accelerator.log({f"eval_ema_gen_{i}": result}, step=global_step)
                dit.load_state_dict(original_state_dict, strict=False)
            
            torch.cuda.empty_cache()
            gc.collect()
            torch.set_grad_enabled(True)
            dit.train()
            accelerator.wait_for_everyone()

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
    
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    parser = transformers.HfArgumentParser([TrainArgs])
    args_tuple = parser.parse_args_into_dataclasses(args_file_flag="--config")
    main(*args_tuple)
