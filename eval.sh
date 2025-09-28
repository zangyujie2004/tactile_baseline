#!/bin/bash

# DP w. GelSight Emb. (Peeling)
#python eval_real_robot_flexiv.py \
#      --config-name train_diffusion_unet_real_image_workspace \
#      task=real_peel_image_gelsight_emb_absolute_12fps \
#      +task.env_runner.output_dir=/path/for/saving/videos \
#      +ckpt_path=/path/to/dp/checkpoint

# RDP w. Force (Peeling)
CUDA_VISIBLE_DEVICES=0 python eval_real_robot_flexiv.py \
      --config-name train_latent_diffusion_unet_real_image_workspace \
      task=real_wipe_image_gelsight_emb_ldp_24fps \
      +task.env_runner.output_dir=/path/for/saving/videos \
      at=at_peel \
      +ckpt_path=/home/pc/workspace/zyh/reactive_diffusion_policy/data/outputs/2025.07.25/22.03.52_train_vae_real_wipe_image_gelsight_emb_at_24fps_07250001/checkpoints/latest.ckpt \
      at_load_dir=/home/pc/workspace/zyh/reactive_diffusion_policy/data/outputs/2025.07.26/16.06.06_train_latent_diffusion_unet_image_real_wipe_image_gelsight_emb_ldp_24fps_07250001/checkpoints/latest.ckpt