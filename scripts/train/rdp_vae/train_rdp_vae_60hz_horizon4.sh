# #!/bin/bash

GPU_ID=2

TASK_NAME="wipe"
# Point to the dataset directory that contains 'replay_buffer.zarr'
# DATASET_PATH="/data/kywang/projects/tactile_il/data/processed/vase_new_C/rdp_zarr"
DATASET_PATH="data/ckpts/vase_sponge_test1_60hz/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=vase_sponge1_rdp_vae_60hz_horizon4
SEARCH_PATH="./data/outputs"

# Stage 1: Train Asymmetric Tokenizer
echo "Stage 1: training Asymmetric Tokenizer..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=train_at_workspace \
    task=real_${TASK_NAME}_image_gelsight_emb_at_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=real_${TASK_NAME}_${TIMESTAMP} \
    at=at_wipe_lift \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=4 \
    at.n_obs_steps=1 \
    at.policy.use_rnn_decoder=False


# # find the latest checkpoint
# echo ""
# echo "Searching for the latest AT checkpoint..."
# AT_LOAD_DIR=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${TIMESTAMP}*" -type d)/checkpoints/latest.ckpt
# # AT_LOAD_DIR=data/outputs/2025.11.13/22.03.48_train_vae_real_wipe_image_gelsight_emb_at_24fps_vase_a/checkpoints/latest.ckpt

# # echo $(find "${SEARCH_PATH}" -maxdepth 2 -path "*${TIMESTAMP}*" -type d)/checkpoints/latest.ckpt
# # echo ${AT_LOAD_DIR}

# # AT_LOAD_DIR="data/outputs/2025.11.26/07.11.14_train_vae_real_wipe_image_gelsight_emb_at_24fps_vase2_new_ABC/checkpoints/latest.ckpt"

# if [ ! -f "${AT_LOAD_DIR}" ]; then
#     echo "Error: VAE checkpoint not found at ${AT_LOAD_DIR}"
#     exit 1
# fi

# # # Stage 2: Train Latent Diffusion Policy
# echo ""
# echo "Stage 2: training Latent Diffusion Policy..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
#     --config-name=train_latent_diffusion_unet_real_image_workspace \
#     task=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps \
#     task.dataset_path=${DATASET_PATH} \
#     task.dataset.relative_action=False \
#     task.name=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps_${TIMESTAMP} \
#     at=at_wipe_lift \
#     at_load_dir=${AT_LOAD_DIR} \
#     logging.mode=${LOGGING_MODE}