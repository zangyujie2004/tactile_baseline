python infer_forcemimic.py \
  --config-name train_forcemimic_image_workspace \
  task=forcemimic \
  task.name=forcemimic_infer \
  logging.mode=offline \
  +load_pca_path=/home/tars/projects/visual_tactile_policy/Tactile-Baseline/ckpt/pca \
  load_ckpt_path=/home/tars/projects/visual_tactile_policy/Tactile-Baseline/ckpt/forcemimic/cut/latest.ckpt