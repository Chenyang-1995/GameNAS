GameNAS#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
export PYTHONPATH=./:$PYTHONPATH




python pyt_GameNAS/train_final_imagenet.py \
    --auxiliary \
    --arch='GameNAS_image' \
    --gpu=0 2>&1




