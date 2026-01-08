# #!/bin/bash

GPU_ID=2

horizon=48

TASK_NAME="wipe"
# Point to the dataset directory that contains 'replay_buffer.zarr'
# DATASET_PATH="/data/kywang/projects/tactile_il/data/processed/vase_new_C/rdp_zarr"
DATASET_PATH="data/ckpts/vase_sponge_test1_60hz/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=vase_sponge1_rdp_ldp_block_vae_60hz_horizon${horizon}
SEARCH_PATH="./data/outputs"



AT_LOAD_DIR="/home/kywang/projects/efficient_robot_sys/data/ckpts/vase_sponge_test1_60hz/ckpts_abs/rdp_block_vae/n_embed_10/horizon${horizon}/latest.ckpt"

# # # Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_block_vae_workspace \
    +policy.change_kernel_size=False \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    task=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.use_block_vae=True \
    task.dataset.at._target_=reactive_diffusion_policy.model.vae.block_vae_model.BlockEncodeVAE \
    +task.dataset.at.encode_horizon=12 \
    +task.dataset.at.encode_block_num=4 \
    task.dataset.relative_action=False \
    task.name=real_${TASK_NAME}_ldp_${TIMESTAMP} \
    at=rdp_block_vae_wipe_lift \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=${horizon} \
    at.encode_horizon=12 \
    at.encode_block_num=4 \
    at.n_obs_steps=1 \
    at.policy.use_rnn_decoder=False