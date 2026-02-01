python infer_dp.py \
  --config-name train_diffusion_unet_real_image_workspace \
  task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
  task.name=dp_infer \
  logging.mode=offline \
  +load_pca_path=/home/tars/projects/visual_tactile_policy/Tactile-Baseline/ckpt/pca \
  load_ckpt_path=/home/tars/projects/visual_tactile_policy/Tactile-Baseline/ckpt/dp_tactile/cut/latest.ckpt
