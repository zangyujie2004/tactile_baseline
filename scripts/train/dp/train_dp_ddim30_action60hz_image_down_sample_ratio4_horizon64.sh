# a#!/bin/bash
n_obs_steps=1
horizon=64


CUDA_VISIBLE_DEVICES=6 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
    task.dataset_path=data/ckpts/vase_sponge_test1_60hz/rdp_zarr \
    task.name=dp_ddim30_60hz_obs1_horizon64 \
    logging.mode=online \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    horizon=${horizon} \
    n_action_steps=${horizon} \
    +task.dataset.image_downsample_ratio=1 \
    +policy.image_downsample_ratio=1 \
    n_obs_steps=${n_obs_steps}

    #training.num_epochs=1000 \
    #dataloader.batch_size=1 \
    #dataloader.num_workers=1 \
    #val_dataloader.batch_size=1 \
    #val_dataloader.num_workers=1