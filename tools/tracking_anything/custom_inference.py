import gradio as gr
import argparse
import gdown
import cv2
import numpy as np
import os
import sys
sys.path.append(sys.path[0]+"/tracker")
sys.path.append(sys.path[0]+"/tracker/model")
from track_anything import TrackingAnything
from track_anything import parse_augment
import requests
import json
import torchvision
import torch 
from tools.painter import mask_painter
import psutil
import time
try: 
    from mmcv.cnn import ConvModule
except:
    os.system("mim install mmcv")

import PIL
from PIL import Image

# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath

if __name__ == "__main__":
    # args, defined in track_anything.py
    args = parse_augment()

    # check and download checkpoints if needed
    SAM_checkpoint_dict = {
        'vit_h': "sam_vit_h_4b8939.pth",
        'vit_l': "sam_vit_l_0b3195.pth", 
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    SAM_checkpoint_url_dict = {
        'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type] 
    sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type] 
    xmem_checkpoint = "XMem-s012.pth"
    xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
    e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
    e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"


    folder ="./checkpoints"
    SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
    xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
    e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)
    args.port = 12212
    args.device = "cuda:0"
    # args.mask_save = True

    # initialize sam, xmem, e2fgvi models
    model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint,args)
    
    folder_path = args.folder_path
    save_path = args.save_path
    template_mask_folder = args.template_mask_folder
    
    os.makedirs(save_path, exist_ok=True)
     
    # process image
    num_images = len(os.listdir(folder_path))
    
    num_images_per_video = num_images // 4 # four view
            

    os.makedirs(save_path, exist_ok=True)
    
    for view_idx in range(4):

        frames = [Image.open(os.path.join(folder_path, f"{view_idx*num_images_per_video+image_idx}.png")) for image_idx in range(num_images_per_video)]
        frames = [frame.convert("RGB").resize((512, 512), PIL.Image.LANCZOS) for frame in frames]
        frames = [np.array(frame) for frame in frames]
        
        template_mask_path = os.path.join(template_mask_folder, f"{view_idx}.png")
        template_mask = Image.open(template_mask_path)
        # template_mask = template_mask.resize((512, 512), Image.ANTIALIAS)
        template_mask = np.array(template_mask)[..., -1] > 255/2
        template_mask = template_mask * 1.
        
        masks, logits, painted_images = model.generator(images=frames, template_mask=template_mask)
        # clear GPU memory
        model.xmem.clear_memory()

        for image_idx in range(num_images_per_video):
            rgb = Image.open(os.path.join(folder_path, f"{view_idx*num_images_per_video+image_idx}.png")).convert("RGB")
            rgb = np.array(rgb)
            mask = masks[image_idx] * 255
            mask = Image.fromarray(mask).resize((256, 256))
            mask = np.array(mask)
            
            save_img = np.concatenate([rgb, mask[:, :, None]], axis=-1)
            save_img = Image.fromarray(save_img)
            save_img.save(os.path.join(save_path, f'{view_idx*num_images_per_video+image_idx}.png'))
            
                