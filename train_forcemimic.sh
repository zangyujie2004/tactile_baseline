#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_forcemimic_image_workspace \
    task=forcemimic \
    task.dataset_path=/home/pc/workspace/tactile_data/board/rdp_zarr \
    task.name=dp_forcemimic_dish \
    logging.mode=online







