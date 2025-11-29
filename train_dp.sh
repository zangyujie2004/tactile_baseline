# a#!/bin/bash

CUDA_VISIBLE_DEVICES=2 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
    task.dataset_path=data/ckpts/vase2_new_A_vase2_new_B_vase2_new_C/rdp_zarr \
    task.name=vase2_new_ABC_dp \
    logging.mode=online \
    #training.num_epochs=1000 \
    #dataloader.batch_size=1 \
    #dataloader.num_workers=1 \
    #val_dataloader.batch_size=1 \
    #val_dataloader.num_workers=1