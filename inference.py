import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, MotionAdapter

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet_motion_mv_model import MVUNetMotionModel
from animatediff.models.unet_mv_model import MVUNet2DConditionModel


from animatediff.models.attention_processor import (
    MVDreamXFormersAttnProcessor, 
    IPAdapterXFormersAttnProcessor, 
    MVDreamI2VXFormersAttnProcessor,
    SpatioTemporalI2VXFormersAttnProcessor
)


from animatediff.pipelines.pipeline import AnimateDiffMVI2VPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print, load_ip_adapter, export_to_gif_mv, IPAdapterImageProcessor

from diffusers.models.attention_processor import IPAdapterAttnProcessor

from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version, load_image

from diffusers.models.lora import LoRALinearLayer

from einops import rearrange, repeat

import csv, pdb, glob
import math
from animatediff.utils.util import export_to_gif_mv

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path


def main(args):
    
    config  = OmegaConf.load(args.config)
    
    savedir = f"{config.output_dir}/{args.save_name}"
    os.makedirs(savedir, exist_ok=True)
    
    # Basic setting
    num_views = args.N
    video_length = args.L
    sample_size = args.W

    ### >>> create validation pipeline >>> ###
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.noise_scheduler_kwargs))

    tokenizer    = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")            
    
    mvunet = MVUNet2DConditionModel.from_pretrained(config.pretrained_model_path, subfolder="unet")

    motion_adapter = MotionAdapter.from_pretrained(config.motion_adapter_path)
    unet = MVUNetMotionModel.from_unet2d(
        mvunet, 
        motion_adapter
    )

    ip_feature_extractor, ip_image_encoder = None, None

    # load ip_adapter
    ip_adapter_state_dict, ip_feature_extractor, ip_image_encoder = load_ip_adapter(
                config.ip_adapter_path, 
                subfolder="models", 
                weight_name="ip-adapter_sd15.bin",
                device=vae.device,
                dtype=vae.dtype,
            )
    unet._load_ip_adapter_weights(ip_adapter_state_dict)
    # initialize ip_image_encoder
    ip_image_processor = IPAdapterImageProcessor(ip_feature_extractor, ip_image_encoder, device=vae.device)
    

    # init adapter modules, copy from train.py
    if config.motion_module_attn_cfg.enabled and (config.motion_module_attn_cfg.spatial_attn.enabled or config.motion_module_attn_cfg.image_attn.enabled):
        # init adapter modules
        sample_size = 256
        feature_height = sample_size // 8
        feature_width = sample_size // 8

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
            if config.motion_module_attn_cfg.enabled:
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
                    spatial_attn=config.motion_module_attn_cfg.spatial_attn, image_attn=config.motion_module_attn_cfg.image_attn, use_alpha_blender=config.motion_module_attn_cfg.use_alpha_blender)

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
            
            if config.mvdream_attn_cfg.image_attn.enabled:
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
    if config.motion_module_attn_cfg.enabled and config.motion_module_attn_cfg.spatial_attn.enabled and       \
        (config.motion_module_attn_cfg.spatial_attn.attn_cfg.use_spatial_encoding or config.motion_module_attn_cfg.spatial_attn.attn_cfg.use_camera_encoding):
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



    # import ipdb
    # ipdb.set_trace()
    assert args.pretrained_unet_path is not None
    pretrained_unet_path = args.pretrained_unet_path
     
        
    ## Start Evaluuation ###
    prompt      = args.prompt
    n_prompt    = ""

    ip_image_name = args.ip_image_name
    ip_image_root = args.ip_image_root

    random_seeds = config.get("seed", [-1])
    random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
    # random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        
    print('load unet weight from: ', pretrained_unet_path)
    unet_checkpoint_loaded = torch.load(pretrained_unet_path, map_location="cpu")
    state_dict = unet_checkpoint_loaded["state_dict"] if "state_dict" in unet_checkpoint_loaded else unet_checkpoint_loaded

    # handle dist: if the model is saved with DistributedDataParallel, remove the prefix
    # state_dict = {key[7:]:value for key, value in state_dict.items()}
    m, u = unet.load_state_dict(state_dict, strict=False)
    
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    assert len(m) == 0 or len(m) == 726, "missing keys should be 0 (save full model) or 726 (save motion modules only)"
    assert len(u) == 0, "file is broken!"
    
    pipeline = AnimateDiffMVI2VPipeline(
        unet=unet, 
        vae=vae, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        scheduler=noise_scheduler, 
        motion_adapter=None,
        feature_extractor=ip_feature_extractor, 
        image_encoder=ip_image_encoder
        ).to("cuda")
    
    # set to eval
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.unet.eval()

    # enable FreeInit
    # Refer to the enable_free_init documentation for a full list of configurable parameters
    # https://github.com/huggingface/diffusers/blob/v0.28.0/src/diffusers/pipelines/free_init_utils.py#L38
    if config.freeinit_enabled:
        pipeline.enable_free_init(method="butterworth", num_iters=config.freeinit_num_iters, use_fast_sampling=False)

    # TODO: test it
    pipeline.enable_vae_slicing()
    
    
    samples = []
    sample_idx = 0
    # config[config_key].random_seed = []
    for random_seed in random_seeds:    
        # manually set random seed for reproduction
        if random_seed != -1: torch.manual_seed(random_seed)
        else: torch.seed()
        # config[config_key].random_seed.append(torch.initial_seed())
        save_name_ = "-".join((prompt.replace("/", "").split(" ")[:10]))
        if os.path.exists(f"{savedir}/{sample_idx}-{save_name_}.gif"):
            sample_idx += 1
            continue
        print(f"current seed: {torch.initial_seed()}")
        print(f"sampling {prompt} ...")
        sample = pipeline(
            prompt,
            negative_prompt     = n_prompt,
            num_inference_steps = config.steps,
            guidance_scale      = config.guidance_scale,
            width               = args.W,
            height              = args.H,
            num_frames        = args.L,
            num_videos_per_prompt = args.N, 
            ip_adapter_image = [load_image(os.path.join(ip_image_root, f"{ip_image_name}_{i}.png")) for i in range(num_views)] if ip_image_name != "" else [load_image(os.path.join(ip_image_root, f"{i}.png")) for i in range(num_views)],
            i2v_cond_time_zero=config.i2v_cond_time_zero,
            i2v_similarity_init=config.i2v_similarity_init,
        ).frames
        samples.append(sample)
        
        # save_name_ = "-".join((prompt.replace("/", "").split(" ")[:10]))
        export_to_gif_mv(sample, f"{savedir}/{sample_idx}-{save_name_}.gif")
        print(f"save to {savedir}/{sample_idx}-{save_name_}.gif")
        
        sample_idx += 1

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_unet_path", type=str, default=None, required=True)
    parser.add_argument("--config",                type=str, required=True)
    parser.add_argument("--save_name",                type=str, default='debug')
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--N", type=int, default=4)
    
    parser.add_argument("--ip_image_root", type=str, default="examples/images")
    parser.add_argument("--ip_image_name", type=str, default="", required=True)
    parser.add_argument("--prompt", type=str, default="", required=True) 

    
    args = parser.parse_args()
    main(args)
