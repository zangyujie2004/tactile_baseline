#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_3dtactile_workspace \
    task=real_wipe_image_3D_tactile \
    task.dataset_path=/home/pc/workspace/tactile_data/board/rdp_zarr \
    task.name=dp_3d_tactile_dish \
    logging.mode=online