#!/bin/bash
nnodes=$1
nproc_per_node=$2
config=$3
torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node train.py --config "configs/training/${config}.yaml" --tensorboard