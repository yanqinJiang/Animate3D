#!/bin/bash
gpu=$1
config=$2

prompt=$3
ip_image_root=$4
ip_image_name=$5
save_name=$6

default_pretrained_unet_path="pretrained_models/animate3d_motion_modules.ckpt"
pretrained_unet_path=${7:-$default_pretrained_unet_path}

export CUDA_VISIBLE_DEVICES=$gpu
python inference.py --config "configs/inference/${config}.yaml" \
    --pretrained_unet_path $pretrained_unet_path \
    --W 256 \
    --H 256 \
    --L 16 \
    --N 4 \
    --ip_image_root "$ip_image_root" \
    --ip_image_name "$ip_image_name" \
    --prompt "a lion is idling" \
    --save_name "$save_name"
