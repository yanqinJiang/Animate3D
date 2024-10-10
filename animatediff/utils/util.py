import os
from pathlib import Path
from typing import List, Dict, Union, Optional

import re

import subprocess

import imageio
import torchvision
import torch.distributed as dist

import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
from PIL import Image
from diffusers.utils import export_to_gif
from safetensors import safe_open

from diffusers.utils import (
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    is_transformers_available,
    logging,
)


from huggingface_hub.utils import validate_hf_hub_args

from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT

if is_transformers_available():
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    from diffusers.models.attention_processor import (
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0,
    )

logger = logging.get_logger(__name__)

# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/ip_adapter.py#L50
@validate_hf_hub_args
def load_ip_adapter(
        pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, torch.Tensor]],
        subfolder: Union[str, List[str]],
        weight_name: Union[str, List[str]],
        image_encoder_folder: Optional[str] = "image_encoder",
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    ):

    # handle the list inputs for multiple IP Adapters
    if not isinstance(weight_name, list):
        weight_name = [weight_name]

    if not isinstance(pretrained_model_name_or_path_or_dict, list):
        pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
    if len(pretrained_model_name_or_path_or_dict) == 1:
        pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict * len(weight_name)

    if not isinstance(subfolder, list):
        subfolder = [subfolder]
    if len(subfolder) == 1:
        subfolder = subfolder * len(weight_name)

    if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
        raise ValueError("`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.")

    if len(weight_name) != len(subfolder):
        raise ValueError("`weight_name` and `subfolder` must have the same length.")

    # Load the main state dict first.
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

    if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

    if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
            " `low_cpu_mem_usage=False`."
        )

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    state_dicts = []
    for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
        pretrained_model_name_or_path_or_dict, weight_name, subfolder
    ):
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["image_proj", "ip_adapter"]:
            raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

        state_dicts.append(state_dict)

            # load CLIP image encoder here if it has not been registered to the pipeline yet

    if image_encoder_folder is not None:
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            logger.info(f"loading image_encoder from {pretrained_model_name_or_path_or_dict}")
            if image_encoder_folder.count("/") == 0:
                image_encoder_subfolder = Path(subfolder, image_encoder_folder).as_posix()
            else:
                image_encoder_subfolder = Path(image_encoder_folder).as_posix()

            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                pretrained_model_name_or_path_or_dict,
                subfolder=image_encoder_subfolder,
                low_cpu_mem_usage=low_cpu_mem_usage,
            ).to(device, dtype=dtype)
        else:
            raise ValueError(
                "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
            )

    feature_extractor = CLIPImageProcessor()
    # # load ip-adapter into unet
    # self.unet._load_ip_adapter_weights(state_dict)

    return state_dict, feature_extractor, image_encoder

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def export_to_gif_mv(samples, save_path):
    num_images = len(samples[0])
    # 用于保存横向拼接后的图像的列表
    concat_images = []

    # 遍历每个图像索引
    for i in range(num_images):
        # 提取每个子列表中的第i个图像
        images_to_concat = [sublist[i] for sublist in samples if i < len(sublist)]

        # 获取所有图像的宽度和高度
        widths, heights = zip(*(i.size for i in images_to_concat))
        
        # 计算拼接后的总宽度和最大高度
        total_width = sum(widths)
        max_height = max(heights)
        
        # 创建一个新图像，它的宽度是所有图像宽度之和，高度是所有图像中的最大高度
        new_im = Image.new('RGB', (total_width, max_height))
        
        # 横向拼接图像
        x_offset = 0
        for im in images_to_concat:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        
        # 将拼接后的图像添加到结果列表中
        concat_images.append(new_im)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    export_to_gif(concat_images, save_path)


def find_latest_checkpoint(folder_path):
    # 初始化最大的尾缀数字以及相应的文件名
    max_suffix = -1
    last_ckpt_file = None

    # 正则表达式，用于匹配文件名并提取数字尾缀
    pattern = re.compile(r'checkpoint-epoch-(\d+)\.ckpt')

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 使用正则表达式搜索匹配
        match = pattern.search(filename)
        if match:
            # 提取尾缀数字
            suffix_number = int(match.group(1))
            # 更新最大尾缀和相应的文件名
            if suffix_number > max_suffix:
                max_suffix = suffix_number
                last_ckpt_file = filename

    # 返回最后一个ckpt文件的名称
    return last_ckpt_file

def get_git_commit_id(repo_path):
    """
    Get the latest git commit ID of the repository at the given path.

    Parameters:
    repo_path (str): The path to the repository.

    Returns:
    str: The commit ID if the repository exists and is a git repository, otherwise None.
    """
    # Check if the .git directory exists
    if not os.path.isdir(os.path.join(repo_path, '.git')):
        return None
    
    try:
        # Run 'git rev-parse HEAD' to get the commit ID
        commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo_path)
        return commit_id.decode('utf-8').strip()  # Decode bytes to string and strip newlines
    except subprocess.CalledProcessError as e:
        return None

# Modified from https://github.com/huggingface/diffusers/blob/v0.24.0/src/diffusers/pipelines/animatediff/pipeline_animatediff.py#L323
class IPAdapterImageProcessor(nn.Module):
    def __init__(self, feature_extractor, image_encoder, device="cuda"):
        super(IPAdapterImageProcessor, self).__init__()  # 初始化父类的 __init__
        
        self.feature_extractor = feature_extractor
        self.image_encoder = image_encoder
        
    def encode_image(self, image_tensor_list):
        # tensor should be between [0, 1.] or [0, 255]
        dtype = next(self.image_encoder.parameters()).dtype
        device = next(self.image_encoder.parameters()).device

        
        image = self.feature_extractor(image_tensor_list, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        # image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return image_embeds
