#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
export PYTHONPATH=./:$PYTHONPATH




python pyt_GameNAS/train_search.py \
            --max_arc_size=3.2 \
            --num_arc_class=3 \
            --epochs=150 \
            --num_iterations=150 2>&1