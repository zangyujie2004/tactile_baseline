python infer_dp_reverse.py \
  --config-name train_diffusion_unet_real_image_workspace_reverse \
  task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps_reverse \
  task.name=dp_infer \
  logging.mode=offline \
  +load_pca_path=/home/tars/projects/dataset/vase2_new_A/rdp_pca \
  load_ckpt_path=data/outputs/2025.11.17/20.57.34_train_diffusion_unet_image_dp_tactile_vase/checkpoints/latest.ckpt
