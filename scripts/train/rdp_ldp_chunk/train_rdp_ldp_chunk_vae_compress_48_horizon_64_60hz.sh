# #!/bin/bash

GPU_ID=0

TASK_NAME="wipe"
# Point to the dataset directory that contains 'replay_buffer.zarr'
# DATASET_PATH="/data/kywang/projects/tactile_il/data/processed/vase_new_C/rdp_zarr"
DATASET_PATH="data/ckpts/vase_sponge_test1_60hz/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=60hz_compress64
SEARCH_PATH="./data/outputs"

# Stage 1: Train Asymmetric Tokenizer
# echo "Stage 1: training Asymmetric Tokenizer..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
#     --config-name=train_at_workspace \
#     task=real_${TASK_NAME}_image_gelsight_emb_at_24fps \
#     task.dataset_path=${DATASET_PATH} \
#     task.dataset.relative_action=False \
#     task.name=real_${TASK_NAME}_image_gelsight_emb_at_24fps_${TIMESTAMP} \
#     at=at_wipe_lift \
#     logging.mode=${LOGGING_MODE} \
#     at.dataset_obs_temporal_downsample_ratio=1 \
#     at.horizon=32 \
#     at.n_obs_steps=1 \
#     at.policy.use_rnn_decoder=False



AT_LOAD_DIR="data/ckpts/vase_sponge_test1_60hz/ckpts_abs/rdp_vae/horizon64/latest.ckpt"

vae_compress_ratio=64
at_horizon=64
ldp_horizon=64 # total horizon, ldp_latent_horizon is ldp_horizon//at_horizon

# # # Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_chunk_diffusion_unet_real_image_workspace \
    task=real_${TASK_NAME}_image_gelsight_emb_ldp_chunk \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=real_${TASK_NAME}_ldp_chunk_${TIMESTAMP} \
    at=at_wipe_lift \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    horizon=${ldp_horizon} \
    at.horizon=${at_horizon} \
    at.policy.horizon=${at_horizon} \
    at.n_obs_steps=1 \
    at.policy.use_rnn_decoder=False \
    policy.vae_compress_ratio=${vae_compress_ratio} \
    task.dataset.vae_compress_ratio=${vae_compress_ratio}