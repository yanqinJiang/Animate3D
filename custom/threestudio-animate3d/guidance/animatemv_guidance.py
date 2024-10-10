from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, MotionAdapter
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL

from animatediff.models.unet_motion_mv_model import MVUNetMotionModel
from animatediff.models.unet_mv_model import MVUNet2DConditionModel

from animatediff.models.attention_processor import (
    MVDreamXFormersAttnProcessor, 
    IPAdapterXFormersAttnProcessor, 
    MVDreamI2VXFormersAttnProcessor,
    SpatioTemporalI2VXFormersAttnProcessor
)
from diffusers.models.attention_processor import IPAdapterAttnProcessor
from animatediff.pipelines.pipeline import AnimateDiffMVI2VPipeline

from animatediff.pipelines.pipeline import tensor2vid
from animatediff.utils.util import load_ip_adapter, IPAdapterImageProcessor

import numpy as np
from PIL import Image
from einops import rearrange

# copied from mvdream/camera_utils.py
def normalize_camera(camera_matrix):
    ''' normalize the camera location onto a unit-sphere'''
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    return camera_matrix.reshape(-1,16)

@threestudio.register("animatemv-diffusion-guidance")
class AnimateMVDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        # animatemv
        pretrained_model_name_or_path: str = ""
        motion_adapter_path: str = None
        ip_adapter_path: str = None
        pretrained_unet_path: str = None

        model_config: dict = None
        
        # disable options that will change attn_procs
        # enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        # enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        sqrt_anneal: bool = False  # sqrt anneal proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        trainer_max_steps: int = 25000

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        # mvdream + animatediff
        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        # i2v
        i2v_cond_time_zero: bool = False
        n_view: int = 4
        n_frame: int = 8
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5

        # noise scheduler kwargs
        noise_scheduler_kwargs: dict = None

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading AnimateMV Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        
        ############ Animatemv: initialization #################
        tokenizer    = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="text_encoder")
        vae          = AutoencoderKL.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="vae")            
        
        mvunet = MVUNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")

        motion_adapter = MotionAdapter.from_pretrained(self.cfg.motion_adapter_path)
        unet = MVUNetMotionModel.from_unet2d(
            mvunet, 
            motion_adapter
        )

        ip_feature_extractor, ip_image_encoder = None, None

        if self.cfg.ip_adapter_path is not None:
            # load ip_adapter
            ip_adapter_state_dict, ip_feature_extractor, ip_image_encoder = load_ip_adapter(
                        self.cfg.ip_adapter_path, 
                        subfolder="models", 
                        weight_name="ip-adapter_sd15.bin",
                        device=vae.device,
                        dtype=vae.dtype,
                    )
            unet._load_ip_adapter_weights(ip_adapter_state_dict)
            # initialize ip_image_encoder
            self.ip_image_processor = IPAdapterImageProcessor(ip_feature_extractor, ip_image_encoder, device=vae.device)
        

        # init adapter modules, copy from train_i2v.py
        if self.cfg.model_config.motion_module_attn_cfg.enabled and (self.cfg.model_config.motion_module_attn_cfg.spatial_attn.enabled or self.cfg.model_config.motion_module_attn_cfg.image_attn.enabled):
            # init adapter modules
            sample_size = 256 # hard-coded
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
                if self.cfg.model_config.motion_module_attn_cfg.enabled:
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
                    
                    attn_procs[attn_processor_name] = SpatioTemporalI2VXFormersAttnProcessor(hidden_size=hidden_size, feature_size=feature_size, num_views=self.cfg.n_view, num_frames=self.cfg.n_frame, \
                        spatial_attn=self.cfg.model_config.motion_module_attn_cfg.spatial_attn, image_attn=self.cfg.model_config.motion_module_attn_cfg.image_attn, use_alpha_blender=self.cfg.model_config.motion_module_attn_cfg.use_alpha_blender)

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
                
                if self.cfg.model_config.mvdream_attn_cfg.image_attn.enabled:
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
                    xformers_attention_processor = MVDreamI2VXFormersAttnProcessor(hidden_size=hidden_size, num_views=self.cfg.n_view, num_frames=self.cfg.n_frame)
                    attn_procs[attn_processor_name] = xformers_attention_processor
                    attn_procs[attn_processor_name].load_state_dict(weights)
                    attn_procs[attn_processor_name].to(weights["to_q_i2v.weight"])
                else:
                    xformers_attention_processor = MVDreamXFormersAttnProcessor(num_frames=self.cfg.n_frame)
                    attn_procs[attn_processor_name] = xformers_attention_processor
                
        unet.set_attn_processor(attn_procs)

        # set BasicTransformerBlock.pos_embed = None for motion_modules if we use spatial encoding
        if self.cfg.model_config.motion_module_attn_cfg.enabled and self.cfg.model_config.motion_module_attn_cfg.spatial_attn.enabled and       \
            (self.cfg.model_config.motion_module_attn_cfg.spatial_attn.attn_cfg.use_spatial_encoding or self.cfg.model_config.motion_module_attn_cfg.spatial_attn.attn_cfg.use_camera_encoding):
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


        print('load unet weight from: ', self.cfg.pretrained_unet_path)
        unet_checkpoint_path = torch.load(self.cfg.pretrained_unet_path, map_location="cpu")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

        # handle dist
        m, u = unet.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(m) == 0 or len(m) == 726, "missing keys should be 0 (save full model) or 726 (save motion modules only)"
        assert len(u) == 0, "file is broken!"
        
        noise_scheduler = DDIMScheduler(**self.cfg.noise_scheduler_kwargs)
      
        self.pipe = AnimateDiffMVI2VPipeline(
            unet=unet, 
            vae=vae, 
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            scheduler=noise_scheduler, 
            motion_adapter=None,
            feature_extractor=ip_feature_extractor,
            image_encoder=ip_image_encoder,
        ).to(self.device).to(self.weights_dtype)

        self.ip_image_processor = self.ip_image_processor.to(self.device).to(self.weights_dtype)
        

        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()
        self.pipe.unet.eval()
       
        self.ip_image_processor.eval()
            
        ################## AnimateMV load end #########################

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        self.scheduler = noise_scheduler
        # note
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.grad_clip_val: Optional[float] = None
        
        threestudio.info(f"Loaded Multi-view Video Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        camera: Float[Tensor, "..."],
        i2v_cond_time_zero: bool,
        added_cond_kwargs = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype

        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            camera=camera.to(self.weights_dtype),
            added_cond_kwargs={key:val.to(self.weights_dtype) for key, val in added_cond_kwargs.items()},
            i2v_cond_time_zero=i2v_cond_time_zero,
        ).sample.to(input_dtype)

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:

        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_mvdream_recon_loss(
        self,
        latents: Float[Tensor, "B 4 32 32"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera: Float[Tensor, "B 4 4"] = None,
        image_embeds = None,
    ):  

        timestamps = t[:, None].repeat(1, self.cfg.n_view)
        timestamps = rearrange(timestamps, "b n -> (b n)")

        batch_size = elevation.shape[0] // (self.cfg.n_view * self.cfg.n_frame)


        neg_guidance_weights = None
        
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation.reshape(batch_size, self.cfg.n_view, self.cfg.n_frame)[..., 0].reshape(-1), azimuth.reshape(batch_size, self.cfg.n_view, self.cfg.n_frame)[..., 0].reshape(-1), camera_distances.reshape(batch_size, self.cfg.n_view, self.cfg.n_frame)[..., 0].reshape(-1), self.cfg.view_dependent_prompting
        )
        # reshape latents
        latents = rearrange(latents, "(b n f) c h w -> b n c f h w", n=self.cfg.n_view, f=self.cfg.n_frame)
        
        
        first_frame_latents = latents[:, :, :, 0:1].clone()
        rest_latents = latents[:,:,:, 1:]
            
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # IMPORTANT: Don't move latents to torch.no_grad()!!!
            # # reshape latents
            # latents = rearrange(latents, "(b n f) c h w -> b n c f h w", n=self.cfg.n_view, f=self.cfg.n_frame)
            # add noise
          
            noise  = torch.randn_like(rest_latents)
            rest_latents_noisy = self.scheduler.add_noise(rest_latents, noise, t)
            latents_noisy = torch.cat([first_frame_latents, rest_latents_noisy], dim=3)
                
            # latents = rearrange(latents, "b n c f h w -> (b n) c f h w")
            latents_noisy = rearrange(latents_noisy, "b n c f h w -> (b n) c f h w")
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            if camera is not None:
                camera = self.get_camera_cond(camera.reshape(batch_size, self.cfg.n_view, self.cfg.n_frame, 4, 4)[:, :, 0].reshape(batch_size * self.cfg.n_view, 4, 4))
                camera = torch.cat([camera] * 2)
            
            image_embeds = torch.cat([image_embeds, torch.zeros_like(image_embeds)])

            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([timestamps] * 2),
                encoder_hidden_states=text_embeddings,
                camera=camera,
                i2v_cond_time_zero=self.cfg.i2v_cond_time_zero,
                added_cond_kwargs={"image_embeds":image_embeds},
            )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            
            noise_pred_text = rearrange(noise_pred_text, "b c f h w -> (b f) c h w")
            noise_pred_uncond = rearrange(noise_pred_uncond, "b c f h w -> (b f) c h w")
            
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # reshape back
        latents = rearrange(latents, " b n c f h w -> (b n f) c h w")
        latents_noisy = rearrange(latents_noisy, "b c f h w -> (b f) c h w")

        # latents_recon = (latents_noisy - sigma * noise_pred) / alpha
        latents_recon = self.scheduler.step(noise_pred, t, latents_noisy).pred_original_sample
        
        if self.cfg.recon_std_rescale > 0:
            # latents_recon_nocfg = (latents_noisy - sigma * noise_pred_text) / alpha
            latents_recon_nocfg = self.scheduler.step(noise_pred_text, t, latents_noisy).pred_original_sample
            latents_recon_nocfg_reshape = rearrange(latents_recon_nocfg, "(b n f) c h w -> b n f c h w", n=self.cfg.n_view, f=self.cfg.n_frame)
            latents_recon_reshape = rearrange(latents_recon, "(b n f) c h w -> b n f c h w", n=self.cfg.n_view, f=self.cfg.n_frame)
            
    
            latents_recon_reshape = latents_recon_reshape[:, :, 1:]
            latents_recon_nocfg_reshape = latents_recon_nocfg_reshape[:, :, 1:]

            factor = (
                latents_recon_nocfg_reshape.std([1, 2, 3, 4, 5], keepdim=True) + 1e-8
            ) / (latents_recon_reshape.std([1, 2, 3, 4, 5], keepdim=True) + 1e-8)

            latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).squeeze(1).repeat_interleave(self.cfg.n_view*self.cfg.n_frame, dim=0)

            latents_recon = (
                self.cfg.recon_std_rescale * latents_recon_adjust
                + (1 - self.cfg.recon_std_rescale) * latents_recon
            )

     
        latents_recon = rearrange(latents_recon, "(b f) c h w -> b f c h w", f=self.cfg.n_frame)
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=self.cfg.n_frame)
        latents_recon = torch.cat([latents[:, 0:1], latents_recon[:, 1:]], dim=1)
        latents = rearrange(latents, "b f c h w -> (b f) c h w")
        latents_recon = rearrange(latents_recon, "b f c h w -> (b f) c h w")

        # x0-reconstruction loss from Sec 3.2 and Appendix
        loss = (
            0.5
            * F.mse_loss(latents, latents_recon.detach(), reduction="sum")
            / latents.shape[0] * self.cfg.n_frame / (self.cfg.n_frame-1)
        )

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "latents_recon": latents_recon,
        }

        return loss, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
 
        batch_size = rgb.shape[0] // (self.cfg.n_view * self.cfg.n_frame)
        camera = c2w

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_256 = F.interpolate(
            rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (32, 32), mode="bilinear", align_corners=False
            )
        else:
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_256)

        image_embeds = None

        with torch.no_grad():
            cond_image_tensor_list = rearrange(rgb_BCHW, "(b f) c h w -> b f c h w", f=self.cfg.n_frame)[:, 0]
            # cond_image_tensor_list = [image for image in cond_image_tensor_list]
            # image_embeds = self.ip_image_processor.encode_image(cond_image_tensor_list)
            # TODO: check it!
            # directly use tensor foramt as in train.py will results in wrong image_embeds. Strange.
            # The reason is transformers update (4.25.1 to 4.28.1)
            cond_image_list = [image.permute(1, 2, 0).detach().cpu().numpy() for image in cond_image_tensor_list]
            cond_image_list = [Image.fromarray((image*255).astype(np.uint8)) for image in cond_image_list]
            image_embeds = self.ip_image_processor.encode_image(cond_image_list)
        

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        recon_loss, guidance_eval_utils = self.compute_mvdream_recon_loss(
            latents,
            t,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            camera,
            image_embeds,
        )

   
        loss_sds = recon_loss

        guidance_out = {
            "loss_sds": loss_sds,
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:

            guidance_eval_out = self.guidance_eval(camera=camera, image_embeds=image_embeds, **guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
        camera=None,
        i2v_cond_time_zero=False,
        image_embeds=None,
    ):
        batch_size = latents_noisy.shape[0] // (self.cfg.n_view * self.cfg.n_frame)
        
        image_embeds = torch.cat([image_embeds, torch.zeros_like(image_embeds)])
            
        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:

            # pred noise
            latents_noisy = rearrange(latents_noisy, "(b f) c h w -> b c f h w", f=self.cfg.n_frame)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            camera = self.get_camera_cond(camera.reshape(batch_size, self.cfg.n_view, self.cfg.n_frame, 4, 4)[:, :, 0].reshape(batch_size * self.cfg.n_view, 4, 4))
            camera = torch.cat([camera]*2)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * self.cfg.n_view * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
                camera=camera,
                i2v_cond_time_zero=i2v_cond_time_zero,
                added_cond_kwargs={"image_embeds":image_embeds},
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        camera,
        image_embeds,
        t_orig,
        text_embeddings,
        latents_noisy,
        latents_recon,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):  

        # use only 25 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(25)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = latents_noisy.shape[0]

        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]
       
        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        latents_noisy_reshaped = rearrange(latents_noisy, "(b f) c h w -> b c f h w", f=self.cfg.n_frame)
        imgs_noisy = self.pipe.decode_latents(latents_noisy_reshaped)
        video_noisy = tensor2vid(imgs_noisy, self.pipe.image_processor, output_type="pil")
        latents_recon_reshaped = rearrange(latents_recon, "(b f) c h w -> b c f h w", f=self.cfg.n_frame)
        imgs_recon = self.pipe.decode_latents(latents_recon_reshaped)
        video_recon = tensor2vid(imgs_recon, self.pipe.image_processor, output_type="pil")

        # get prev latent
        latents_1step = []
        pred_1orig = []
      
        # for b in range(bs):
        step_output = self.scheduler.step(
            noise_pred, t[0], latents_noisy, eta=0.
        )

        latents_1step = step_output["prev_sample"]
        pred_1orig = step_output["pred_original_sample"]

        # concat
        latents_1step = rearrange(latents_1step, "(b f) c h w -> b c f h w", f=self.cfg.n_frame)
        pred_1orig = rearrange(pred_1orig, "(b f) c h w -> b c f h w", f=self.cfg.n_frame)
        latents_1step = torch.cat([latents_recon_reshaped[:, :, 0:1], latents_1step[:, :, 1:]], dim=2)
        pred_1orig = torch.cat([latents_recon_reshaped[:, :, 0:1], pred_1orig[:, :, 1:]], dim=2)
        latents_1step = rearrange(latents_1step, "b c f h w -> (b f) c h w")
        pred_1orig = rearrange(pred_1orig, "b c f h w -> (b f) c h w")

        imgs_1step = self.pipe.decode_latents(rearrange(latents_1step, "(b f) c h w -> b c f h w", f=self.cfg.n_frame))
        imgs_1orig = self.pipe.decode_latents(rearrange(pred_1orig, "(b f) c h w -> b c f h w", f=self.cfg.n_frame))
        video_1step = tensor2vid(imgs_1step, self.pipe.image_processor, output_type="pil")
        video_1orig = tensor2vid(imgs_1orig, self.pipe.image_processor, output_type="pil")

        latents_final = []        
        latents = latents_1step
        text_emb = text_embeddings
        neg_guid = neg_guidance_weights if use_perp_neg else None

        for t in tqdm(self.scheduler.timesteps[idxs[0] + 1 :], leave=False):
            # pred noise
            noise_pred = self.get_noise_pred(
                latents, t, text_emb, use_perp_neg, neg_guid, camera, self.cfg.i2v_cond_time_zero, image_embeds
            )

            # get prev latent
            latents = self.scheduler.step(noise_pred, t, latents, eta=0., generator=None)[
                "prev_sample"
            ]
        
            # concat
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=self.cfg.n_frame)
            latents = torch.cat([latents_recon_reshaped[:, :, 0:1], latents[:, :, 1:]], dim=2)
            latents = rearrange(latents, "b c f h w -> (b f) c h w")

        latents_final.append(latents)


        latents_final = torch.cat(latents_final)
        imgs_final = self.pipe.decode_latents(rearrange(latents_final, "(b f) c h w -> b c f h w", f=self.cfg.n_frame))
        video_final = tensor2vid(imgs_final, self.pipe.image_processor, output_type="pil")
        
        return {
            "bs": bs,
            "noise_levels": fracs,
            "video_noisy": video_noisy,
            "video_recon": video_recon,
            "video_1step": video_1step,
            "video_1orig": video_1orig,
            "video_final": video_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.sqrt_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5  # progress percentage
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )
