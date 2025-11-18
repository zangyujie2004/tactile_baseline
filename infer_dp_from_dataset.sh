python infer_dp_from_dataset.py \
  --config-name train_diffusion_unet_real_image_workspace \
  task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
  task.name=dp_infer \
  logging.mode=offline \
  load_ckpt_path=data/outputs/2025.11.17/19.25.05_train_diffusion_unet_image_dp_tactile_vase/checkpoints/latest.ckpt \
  +load_pca_path=/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_pca \
  task.dataset_path=/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_zarr


# reverse
# python infer_dp_from_dataset.py \
#   --config-name train_diffusion_unet_real_image_workspace_reverse \
#   task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps_reverse \
#   task.name=dp_infer \
#   logging.mode=offline \
#   load_ckpt_path=data/outputs/2025.11.17/20.57.34_train_diffusion_unet_image_dp_tactile_vase/checkpoints/latest.ckpt \
#   +load_pca_path=/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_pca \
#   task.dataset_path=/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_zarr

# test overfit
# python infer_dp_from_dataset.py \
#   --config-name train_diffusion_unet_real_image_workspace \
#   task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
#   task.name=dp_infer \
#   logging.mode=offline \
#   load_ckpt_path=data/outputs/2025.11.18/11.22.28_train_diffusion_unet_image_dp_tactile_vase/checkpoints/latest.ckpt \
#   +load_pca_path=/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_pca \
#   task.dataset_path=/data/kywang/projects/tactile_il/data/processed/vase2_new_A/rdp_zarr