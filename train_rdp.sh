# #!/bin/bash

GPU_ID=0

TASK_NAME="wipe"
# Point to the dataset directory that contains 'replay_buffer.zarr'
DATASET_PATH="/home/pc/workspace/tactile_data/dish_vase_board_/rdp_zarr"
LOGGING_MODE="offline"

# TIMESTAMP=$(date +%m%d%H%M%S)/home/pc/workspace/zyh/reactive_diffusion_policy/data/outputs/2025.09.09/19.30.10_train_vae_real_wipe_image_gelsight_emb_at_24fps_0909000/checkpoints/latest_at.ckpt
TIMESTAMP=0924_
SEARCH_PATH="./data/outputs"

# Stage 1: Train Asymmetric Tokenizer
# echo "Stage 1: training Asymmetric Tokenizer..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
#     --config-name=train_at_workspace \
#     task=real_${TASK_NAME}_image_gelsight_emb_at_24fps \
#     task.dataset_path=${DATASET_PATH} \
#     task.name=real_${TASK_NAME}_image_gelsight_emb_at_24fps_${TIMESTAMP} \
#     at=at_wipe_lift \
#     logging.mode=${LOGGING_MODE}

# find the latest checkpoint
echo ""
echo "Searching for the latest AT checkpoint..."
# AT_LOAD_DIR=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${TIMESTAMP}*" -type d)/checkpoints/latest.ckpt
AT_LOAD_DIR="/home/pc/workspace/zyh/reactive_diffusion_policy/data/outputs/2025.09.24/16.42.03_train_vae_real_wipe_image_gelsight_emb_at_24fps_0924_/checkpoints/latest.ckpt"

if [ ! -f "${AT_LOAD_DIR}" ]; then
    echo "Error: VAE checkpoint not found at ${AT_LOAD_DIR}"
    exit 1
fi

# # Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.name=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps_${TIMESTAMP} \
    at=at_wipe_lift \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE}