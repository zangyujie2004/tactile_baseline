python infer_kinedex.py \
  --config-name train_kinedex_image_workspace \
  task=kinedex \
  task.name=kinedex_infer \
  logging.mode=offline \
  +load_pca_path=/home/tars/projects/visual_tactile_policy/Tactile-Baseline/ckpt/pca \
  load_ckpt_path=/home/tars/projects/visual_tactile_policy/Tactile-Baseline/ckpt/kinedex/cut/latest.ckpt