# a#!/bin/bash
n_obs_steps=1

CUDA_VISIBLE_DEVICES=2 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
    task.dataset_path=data/ckpts/vase_sponge_test1_60hz/rdp_zarr \
    task.name=dp_ddim30_60hz_obs1_original_horizon48_inter4 \
    logging.mode=online \
    policy._target_=reactive_diffusion_policy.policy.interpolate_diffusion_unet_image_policy.InterpolateDiffusionUnetImagePolicy \
    +policy.interpolate_ratio=4 \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    horizon=48 \
    n_action_steps=48 \
    +task.dataset.image_downsample_ratio=1 \
    +policy.image_downsample_ratio=1 \
    n_obs_steps=${n_obs_steps}

    #training.num_epochs=1000 \
    #dataloader.batch_size=1 \
    #dataloader.num_workers=1 \
    #val_dataloader.batch_size=1 \
    #val_dataloader.num_workers=1