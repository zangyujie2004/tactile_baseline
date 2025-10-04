CUDA_VISIBLE_DEVICES=0 python dp_infer.py \
  --config-name train_diffusion_unet_real_image_workspace \
  task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps_not \
  task.name=dp_infer \
  logging.mode=offline \
  load_pca_path=/home/robotics/Prometheus/reactive_diffusion_policy/tactile_pca/board \
  load_ckpt_path=/home/robotics/Tactile-Baseline/ckpt/latest_dp_bb.ckpt
