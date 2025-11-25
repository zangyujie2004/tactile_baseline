#!/bin/bash

# DP w. GelSight Emb. (Peeling)
#python eval_real_robot_flexiv.py \
#      --config-name train_diffusion_unet_real_image_workspace \
#      task=real_peel_image_gelsight_emb_absolute_12fps \
#      +task.env_runner.output_dir=/path/for/saving/videos \
#      +ckpt_path=/path/to/dp/checkpoint

# RDP w. Force (Peeling)
python eval_real_robot_flexiv.py \
      --config-name train_latent_diffusion_unet_real_image_workspace \
      task=rdp \
      +task.env_runner.output_dir=/home/robotics/Prometheus/reactive_diffusion_policy/data/outputs/videos \
      at=at_wipe_lift \
      +ckpt_path=data/outputs/ckpts/vase2_new_A/ldp/latest.ckpt \
      at_load_dir=data/outputs/ckpts/vase2_new_A/at/latest.ckpt \
      pca_load_dir=/home/tars/projects/dataset/vase2_new_A/rdp_pca \

      # task.env_runner.tcp_action_update_interval=4 \
      # task.env_runner.gripper_action_update_interval=4

# vase_c /home/tars/projects/dataset/vase_new_C/rdp_pca
# python eval_real_robot_flexiv.py \
#       --config-name train_latent_diffusion_unet_real_image_workspace \
#       task=rdp \
#       +task.env_runner.output_dir=/home/robotics/Prometheus/reactive_diffusion_policy/data/outputs/videos \
#       at=at_wipe_lift \
#       +ckpt_path= \
#       at_load_dir= \
#       pca_load_dir=/home/tars/projects/dataset/vase_new_C/rdp_pca \