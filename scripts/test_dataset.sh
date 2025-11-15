# 输出dataset的内容, 用于DEBUG

export PYTHONPATH="."

# python tests/test_dataset.py \
#     --config-name train_latent_diffusion_unet_real_image_workspace \
#     task=real_wipe_image_gelsight_emb_ldp_24fps \
#     task.dataset_path="/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_zarr" \
#     dataloader.batch_size=1 \
#     dataloader.shuffle=False 

python tests/test_dataset.py \
      --config-name train_latent_diffusion_unet_real_image_workspace \
      task=rdp \
      task.env_runner._target_=reactive_diffusion_policy.env_runner.real_runner_sync_from_dataset.RealRunner \
      +task.env_runner.output_dir=/home/robotics/Prometheus/reactive_diffusion_policy/data/outputs/videos \
      at=at_wipe_lift \
      +ckpt_path=data/outputs/2025.11.13/12.42.05_train_latent_diffusion_unet_image_real_wipe_image_gelsight_emb_ldp_24fps_vase_a/checkpoints/latest.ckpt \
      at_load_dir=data/outputs/2025.11.13/12.05.26_train_vae_real_wipe_image_gelsight_emb_at_24fps_vase_a/checkpoints/latest.ckpt \
      pca_load_dir=/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_pca \
      task.dataset_path="/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_zarr"