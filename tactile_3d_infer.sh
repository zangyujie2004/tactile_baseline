python vitac_3D_infer.py \
  --config-name train_diffusion_unet_real_3dtactile_workspace \
  task=real_wipe_image_3D_tactile \
  task.name=3d_infer \
  logging.mode=offline \
  load_ckpt_path=/home/robotics/Prometheus/reactive_diffusion_policy/checkpoints/3d_dish_latest.ckpt