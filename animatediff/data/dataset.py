import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

import json
from PIL import Image

from einops import rearrange, repeat

from animatediff.pipelines.pipeline import get_camera


class MVideoDataset(Dataset):
    def __init__(
            self,
            info_path,
            sample_size=256, 
            sample_n_frames=16,
            num_images_per_animation=48,
        ):

        with open(info_path, 'r') as file_to_read:
            data_info = json.load(file_to_read)
        
        self.dataset = data_info
        self.sample_size = sample_size
        self.sample_n_frames = sample_n_frames
        self.num_images_per_animation = num_images_per_animation

        self.sample_stride   = num_images_per_animation // sample_n_frames
   
        self.length = len(self.dataset)
        self.num_views = len(self.dataset[0]['data_path'])

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    
    def __getitem__(self, idx):
        
        data_list = self.dataset[idx]['data_path']
        text_prompt = self.dataset[idx]['text_prompt']
        angle = self.dataset[idx]['angle']
        elv = angle['elv']
        azi_start = angle['azi_start'] # degree

        mv_pixel_values = [] 
      
        start_idx   = random.randint(0, self.sample_stride-1) # random.randint include upper value
        batch_index = np.array([i for i in range(start_idx, self.num_images_per_animation, self.sample_stride)])
        for data_path in data_list:

            video_reader = VideoReader(data_path, width=self.sample_size, height=self.sample_size)
            pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()) # [num_frames, height, width, channel]

            mv_pixel_values.append(pixel_values)
            del video_reader

        mv_pixel_values = torch.stack(mv_pixel_values, dim=0) # [num_view, num_frames, h, w, c]

        pixel_values = mv_pixel_values.permute(0, 1, 4, 2, 3).contiguous() # [num_view, num_frames, c, h, w]
        pixel_values = pixel_values.float() / 255. # [0, 1]
        
        pixel_values = rearrange(pixel_values, "n f c h w -> (n f) c h w")
        pixel_values = self.pixel_transforms(pixel_values) # [-1, 1]
        pixel_values = rearrange(pixel_values, "(n f) c h w -> n f c h w", n=self.num_views)

        name = text_prompt
        
        camera=get_camera(
            self.num_views, 
            elevation=elv, 
            azimuth_start=azi_start,
            azimuth_span=360,
        )
        

        sample = dict(
            cameras=camera,
            pixel_values=pixel_values,
            text=name,
        )

        return sample

    def __len__(self):
        return self.length
