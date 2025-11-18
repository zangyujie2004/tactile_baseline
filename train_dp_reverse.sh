# a#!/bin/bash

CUDA_VISIBLE_DEVICES=3 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace_reverse \
    task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps_reverse \
    task.dataset_path=/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_zarr \
    task.name=dp_tactile_vase \
    logging.mode=offline \
    # training.num_epochs=1 \