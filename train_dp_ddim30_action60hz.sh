# a#!/bin/bash

CUDA_VISIBLE_DEVICES=2 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
    task.dataset_path=data/ckpts/vase_sponge_test1_60hz/rdp_zarr \
    task.name=vase_sponge_test1_dp_ddim30_60hz \
    logging.mode=online \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    horizon=32 \
    n_action_steps=32 \
    +task.dataset.image_downsample_ratio=4
    #training.num_epochs=1000 \
    #dataloader.batch_size=1 \
    #dataloader.num_workers=1 \
    #val_dataloader.batch_size=1 \
    #val_dataloader.num_workers=1