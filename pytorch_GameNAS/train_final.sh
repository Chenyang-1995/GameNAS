GameNAS#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
export PYTHONPATH=./:$PYTHONPATH




python pytorch_GameNAS/train_final.py \
    --auxiliary \
    --arch='GameNAS' \
    --gpu=0 \
    --seed=0 \
    --cutout 2>&1




