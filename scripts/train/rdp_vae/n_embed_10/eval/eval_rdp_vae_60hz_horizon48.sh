# #!/bin/bash

GPU_ID=1

TASK_NAME="wipe"
# Point to the dataset directory that contains 'replay_buffer.zarr'
# DATASET_PATH="/data/kywang/projects/tactile_il/data/processed/vase_new_C/rdp_zarr"
DATASET_PATH="data/ckpts/vase_sponge_test1_60hz/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=vase_sponge1_rdp_vae_60hz_horizon48
SEARCH_PATH="./data/outputs"

# Stage 1: Train Asymmetric Tokenizer
echo "Stage 1: training Asymmetric Tokenizer..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=eval_at_workspace \
    task=real_${TASK_NAME}_image_gelsight_emb_at_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=real_${TASK_NAME}_${TIMESTAMP} \
    at=at_wipe_lift \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=48 \
    at.n_obs_steps=1 \
    at.policy.use_rnn_decoder=False \
    at.policy.n_embed=10 \
    vae_load_path="/home/kywang/projects/efficient_robot_sys/data/ckpts/vase_sponge_test1_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt" \
    dataloader.batch_size=128