a#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
    task.dataset_path=/home/pc/workspace/tactile_data/dish/rdp_zarr \
    task.name=dp_tactile_dish \
    logging.mode=online