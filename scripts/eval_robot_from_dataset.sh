#!/bin/bash
export PYTHONPATH="."

## relative版本的权重, 注意把rdp.yaml改为relative版本的
python eval_robot_from_dataset.py \
      --config-name train_latent_diffusion_unet_real_image_workspace \
      task=rdp \
      task.env_runner._target_=reactive_diffusion_policy.env_runner.real_runner_sync_from_dataset.RealRunner \
      +task.env_runner.output_dir=/home/robotics/Prometheus/reactive_diffusion_policy/data/outputs/videos \
      at=at_wipe_lift \
      +ckpt_path=data/outputs/2025.11.13/22.29.35_train_latent_diffusion_unet_image_real_wipe_image_gelsight_emb_ldp_24fps_vase_a/checkpoints/latest.ckpt \
      at_load_dir=data/outputs/2025.11.13/22.03.48_train_vae_real_wipe_image_gelsight_emb_at_24fps_vase_a/checkpoints/latest.ckpt \
      pca_load_dir=/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_pca \
      task.dataset_path="/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_zarr"
      # task.env_runner.tcp_action_update_interval=4 \
      # task.env_runner.gripper_action_update_interval=4