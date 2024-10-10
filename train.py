import os
import math
import wandb
from torch.utils.tensorboard import SummaryWriter
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, MotionAdapter
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, load_image
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

# from animatediff.data.dataset import WebVid10M
from animatediff.data.dataset import MVideoDataset
from animatediff.models.unet_motion_mv_model import MVUNetMotionModel
from animatediff.models.unet_mv_model import MVUNet2DConditionModel
from animatediff.models.attention_processor import (
    MVDreamXFormersAttnProcessor, 
    IPAdapterXFormersAttnProcessor, 
    MVDreamI2VXFormersAttnProcessor,
    SpatioTemporalI2VXFormersAttnProcessor,
)

from animatediff.pipelines.pipeline import AnimateDiffMVI2VPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print, load_ip_adapter, export_to_gif_mv, IPAdapterImageProcessor, find_latest_checkpoint

import time

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank



def main(
    name: str,
    use_wandb: bool,
    use_tensorboard: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    resume_from_checkpoint: bool=False,
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    motion_adapter_path: str = "",

    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_epoch: int = 5,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
    video_length: int = 8,
    num_views: int = 4,

    ip_adapter_path: str = None,
    mvdream_attn_cfg: dict = None,
    motion_module_attn_cfg: dict = None,

    i2v_cond_time_zero: bool = False,
    
    auto_resume: bool = False,
):
    check_min_version("0.28.0.dev0")

    assert ip_adapter_path is not None, "i2v must be used with ip_adapter!"
    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name # + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
        
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())
  
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    if is_main_process and use_tensorboard:
        writer = SummaryWriter(f"{output_dir}/tf")

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    # import ipdb
    # ipdb.set_trace()
    mvunet = MVUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    motion_adapter = MotionAdapter.from_pretrained(motion_adapter_path)
    unet = MVUNetMotionModel.from_unet2d(
        mvunet, 
        motion_adapter
    )
    ip_feature_extractor, ip_image_encoder = None, None

    # TODO: set ip_adapter scale
    if ip_adapter_path is not None:
        ip_adapter_state_dict, ip_feature_extractor, ip_image_encoder = load_ip_adapter(
            ip_adapter_path, 
            subfolder="models", 
            weight_name="ip-adapter_sd15.bin",
            device=vae.device,
            dtype=vae.dtype,
        )
        unet._load_ip_adapter_weights(ip_adapter_state_dict)
        # initialize ip_image_encoder
        ip_image_processor = IPAdapterImageProcessor(ip_feature_extractor, ip_image_encoder, device=vae.device)

    # init adapter modules
    if motion_module_attn_cfg.enabled and (motion_module_attn_cfg.spatial_attn.enabled or motion_module_attn_cfg.image_attn.enabled):
        # init adapter modules
        feature_height = train_data.sample_size // 8
        feature_width = train_data.sample_size // 8

        num_downsampling_steps = len(unet.down_blocks)
        num_upsampling_steps = len(unet.up_blocks)

        downsampled_sizes = [feature_height]

        for _ in range(num_downsampling_steps-1):
            feature_height //= 2
            feature_width //= 2
            downsampled_sizes.append(feature_height)
    
    attn_procs = {}
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        if "motion_modules" in attn_processor_name:
            # attn_procs[attn_processor_name] = attn_processor
            if motion_module_attn_cfg.enabled:
                if attn_processor_name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                
                    feature_size = downsampled_sizes[-1]
                elif attn_processor_name.startswith("up_blocks"):
                    block_id = int(attn_processor_name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                
                    feature_size = downsampled_sizes[-(block_id + 1)]

                elif attn_processor_name.startswith("down_blocks"):
                    block_id = int(attn_processor_name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
    
                    feature_size = downsampled_sizes[block_id]
                
                attn_procs[attn_processor_name] = SpatioTemporalI2VXFormersAttnProcessor(hidden_size=hidden_size, feature_size=feature_size, num_views=num_views, num_frames=video_length, \
                    spatial_attn=motion_module_attn_cfg.spatial_attn, image_attn=motion_module_attn_cfg.image_attn, use_alpha_blender=motion_module_attn_cfg.use_alpha_blender)
               
            else:
                attn_procs[attn_processor_name] = attn_processor

        elif type(attn_processor) == IPAdapterAttnProcessor:
            # layer_name = attn_processor_name.split(".processor")[0]
            weights = {
                "to_k_ip.0.weight": attn_processor.to_k_ip[0].weight,
                "to_v_ip.0.weight": attn_processor.to_v_ip[0].weight,
            }

            ipadapter_xformers_attention_processor = IPAdapterXFormersAttnProcessor(
                hidden_size=attn_processor.hidden_size, 
                cross_attention_dim=attn_processor.cross_attention_dim, 
                num_tokens=attn_processor.num_tokens, 
                scale=attn_processor.scale
            )
            attn_procs[attn_processor_name] = ipadapter_xformers_attention_processor
            attn_procs[attn_processor_name].load_state_dict(weights)
            attn_procs[attn_processor_name].to(weights["to_k_ip.0.weight"])

        else:
            
            if mvdream_attn_cfg.image_attn.enabled:
                # Parse the attention module. 
                attn_module = unet
                for n in attn_processor_name.split(".")[:-1]:
                    attn_module = getattr(attn_module, n)

                hidden_size = attn_module.to_out[0].out_features
                # I2V weight, skip the dropout layer since p=0.0
                weights = {
                    "to_q_i2v.weight": attn_module.to_q.weight,
                    "to_out_i2v.weight": torch.zeros_like(attn_module.to_out[0].weight), # zero_init
                    "to_out_i2v.bias": torch.zeros_like(attn_module.to_out[0].bias) # zero_init
                }
                xformers_attention_processor = MVDreamI2VXFormersAttnProcessor(hidden_size=hidden_size, num_views=num_views, num_frames=video_length)
                attn_procs[attn_processor_name] = xformers_attention_processor
                attn_procs[attn_processor_name].load_state_dict(weights)
                attn_procs[attn_processor_name].to(weights["to_q_i2v.weight"])
            else:
                xformers_attention_processor = MVDreamXFormersAttnProcessor(num_frames=video_length)
                attn_procs[attn_processor_name] = xformers_attention_processor
            
    unet.set_attn_processor(attn_procs)

    # set BasicTransformerBlock.pos_embed = None for motion_modules if we use spatial encoding
    if motion_module_attn_cfg.enabled and motion_module_attn_cfg.spatial_attn.enabled and       \
        (motion_module_attn_cfg.spatial_attn.attn_cfg.use_spatial_encoding or motion_module_attn_cfg.spatial_attn.attn_cfg.use_camera_encoding):
        # down_block
        num_down_block = len(unet.down_blocks)
        for i in range(num_down_block):
            for j in range(2): # num_motion_modules in down_block
                unet.down_blocks[i].motion_modules[j].transformer_blocks[0].pos_embed = None
        
        # mid_block
        unet.mid_block.motion_modules[0].transformer_blocks[0].pos_embed = None

        # up_block
        num_up_block = len(unet.up_blocks)
        for i in range(num_up_block):
            for j in range(3): # num_motion_modules in up_block
                unet.up_blocks[i].motion_modules[j].transformer_blocks[0].pos_embed = None

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

        # handle dist
        state_dict = {key[7:]:value for key, value in state_dict.items()}
        m, u = unet.load_state_dict(state_dict, strict=False)

        if resume_from_checkpoint:
            resume_epoch, resume_step = unet_checkpoint_path["epoch"], unet_checkpoint_path["global_step"]
            resume_optimizer, resume_scheduler = unet_checkpoint_path["optimizer"], unet_checkpoint_path["lr_scheduler"]

        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        # assert motion_adapter_path == "", "motion_adapter_path will overide unet path!"

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if ip_adapter_path is not None:
        ip_image_processor.requires_grad_(False)

    # Set unet trainable parameters
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break
            
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
 
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if unet_checkpoint_path != "" and resume_from_checkpoint:
        optimizer.load_state_dict(resume_optimizer)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
    if ip_adapter_path is not None:
        ip_image_processor.to(local_rank)

    # Get the training dataset
    train_dataset = MVideoDataset(**train_data)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True, # 保持工作进程活跃
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
    
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)
        auto_resume_checkpointing_steps = len(train_dataloader)
        
    # get validation_steps
    if validation_steps == -1:
        assert validation_epoch != -1
        validation_steps = validation_epoch * len(train_dataloader)
        
    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        last_epoch = resume_epoch if (unet_checkpoint_path != "" and resume_from_checkpoint) else -1,
    )

    if unet_checkpoint_path != "" and resume_from_checkpoint:
        lr_scheduler.load_state_dict(resume_scheduler)
        
    validation_pipeline = AnimateDiffMVI2VPipeline(
        unet=unet, 
        vae=vae, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        scheduler=noise_scheduler, 
        motion_adapter=None,
        feature_extractor=ip_feature_extractor, 
        image_encoder=ip_image_encoder
    ).to("cuda")
    
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = resume_step if (unet_checkpoint_path != "" and resume_from_checkpoint) else 0
    first_epoch = resume_epoch+1 if (unet_checkpoint_path !="" and resume_from_checkpoint) else 0
    if unet_checkpoint_path != "" and resume_from_checkpoint:
        logging.info(f"  Start from epoch {first_epoch} globalstep {global_step}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        # TODO:
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                texts_ = [[t] * num_views for t in texts]
                texts = []
                for t_ in texts_:
                    texts += t_

                pixel_values = rearrange(pixel_values, "b n f c h w -> (b n) c f h w")
                # print(global_rank, batch['text'])
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{text[:100] if not text == '' else f'{global_rank}-{idx}'}_{idx}.gif", rescale=True)
                    
            ### >>>> Training >>>> ###
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[2]
            cameras = batch["cameras"].to(local_rank)

       
            with torch.no_grad():
                    
                if ip_adapter_path is not None:
                    tmp_image_tensor = rearrange(pixel_values[:, :, 0], "b n c h w -> (b n) c h w") # only first frame
                    image_tensor_list = [tmp/2+0.5 for tmp in tmp_image_tensor] # [0, 1.]
                    image_embeds = ip_image_processor.encode_image(image_tensor_list)
                    added_cond_kwargs = {"image_embeds": image_embeds}

                pixel_values = rearrange(pixel_values, "b n f c h w -> (b n f) c h w")

                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()
                latents = rearrange(latents, "(b n f) c h w -> b n c f h w", f=video_length, n=num_views)
                cameras = rearrange(batch["cameras"], "b n c -> (b n) c")

                latents = latents * 0.18215
            
            # I2V: Set the first frame as no noise!!
            # latents: [ b n c f h w ]
            first_frame_latents = latents[:, :, :, 0:1].clone()
            rest_latents = latents[:, :, :, 1:]
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(rest_latents)
            bsz = rest_latents.shape[0]
       
            
            # Sample a random timestsep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
    
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_rest_latents = noise_scheduler.add_noise(rest_latents, noise, timesteps)
            # pesudo noisy _latents
            noisy_latents = torch.cat([first_frame_latents, noisy_rest_latents], dim=3)
            
            noisy_latents = rearrange(noisy_latents, "b n c f h w -> (b n) c f h w")

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
            
            encoder_hidden_states = encoder_hidden_states[:, None].repeat(1, num_views, 1, 1)
            encoder_hidden_states = rearrange(encoder_hidden_states, "b n q c -> (b n) q c")
    
            timesteps = timesteps[:, None].repeat(1, num_views)
            timesteps = rearrange(timesteps, "b n -> (b n)")
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, camera=cameras, num_views=num_views,
                    added_cond_kwargs=added_cond_kwargs if ip_adapter_path is not None else None, i2v_cond_time_zero=i2v_cond_time_zero).sample

                model_pred = rearrange(model_pred, "(b n) c f h w -> b n c f h w", n=num_views)
            
                model_pred_rest_frames = model_pred[:, :, :, 1:] # first frame out
                loss = F.mse_loss(model_pred_rest_frames.float(), target.float(), reduction="mean")
                    
            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
            
            if is_main_process and use_tensorboard:
                writer.add_scalar('Training/Loss', loss.item(), global_step)

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or (global_step % auto_resume_checkpointing_steps == 0)):
                # logging.info(f'global step {global_step}, checkpointing step {checkpointing_steps}, step {step}, length of train_dataloader: {len(train_dataloader)}' )
                
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
            
                if auto_resume:
                    if (global_step % auto_resume_checkpointing_steps == 0):
                        save_path = os.path.join(output_dir, f"checkpoints")
                        state_dict = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "state_dict": unet.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        }
                        torch.save(state_dict, os.path.join(save_path, f"latest.ckpt"))
                        
            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                # samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts[:2] if global_step < 650 else validation_data.prompts
                if ip_adapter_path is not None:
                    ip_images_root = validation_data.image_root
                    ip_images = validation_data.images[:2] if global_step < 650 else validation_data.images
                    
                for idx, prompt in enumerate(prompts):
              
                    sample = validation_pipeline(
                        prompt,
                        generator    = generator,
                        num_frames = train_data.sample_n_frames,
                        height       = height,
                        width        = width,
                        num_videos_per_prompt=num_views,
                        num_inference_steps = validation_data.num_inference_steps,
                        guidance_scale = validation_data.guidance_scale,
                        ip_adapter_image = [load_image(os.path.join(ip_images_root, f"{ip_images[idx]}_{i}.png")) for i in range(num_views)] if ip_adapter_path is not None else None,
                        i2v_cond_time_zero=i2v_cond_time_zero,
                    ).frames
                    
                    export_to_gif_mv(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",       action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    main(launcher=args.launcher, use_wandb=args.wandb, use_tensorboard=args.tensorboard, **config)
