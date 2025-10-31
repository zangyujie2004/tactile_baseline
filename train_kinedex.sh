#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_kinedex_image_workspace \
    task=kinedex \
    task.dataset_path=/home/pc/workspace/tactile_data/board/rdp_zarr \
    task.name=dp_kinedex_dish \
    logging.mode=online







