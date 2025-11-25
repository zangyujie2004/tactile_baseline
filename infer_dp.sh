python infer_dp.py \
  --config-name train_diffusion_unet_real_image_workspace \
  task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
  task.name=dp_infer \
  logging.mode=offline \
  +load_pca_path=/home/tars/projects/pca/vase_all/rdp_pca \
  load_ckpt_path=data/outputs/reverse_ckpts/dp.ckpt
