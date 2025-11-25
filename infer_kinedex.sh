python infer_kinedex.py \
  --config-name train_kinedex_image_workspace \
  task=kinedex \
  task.name=kinedex_infer \
  logging.mode=offline \
  +load_pca_path=/home/tars/projects/pca/vase_all/rdp_pca \
  load_ckpt_path=/home/tars/projects/checkpoints/kinedex_vase_all.ckpt