# #!/bin/bash

GPU_ID=1

TASK_NAME="wipe"
# Point to the dataset directory that contains 'replay_buffer.zarr'
# DATASET_PATH="/data/kywang/projects/tactile_il/data/processed/vase_new_C/rdp_zarr"
DATASET_PATH="data/ckpts/vase_sponge_test1_60hz/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=vase_sponge1_60hz_vae_unet
SEARCH_PATH="./data/outputs"
# num_epochs=1001
num_epochs=1001

# Stage 1: Train Asymmetric Tokenizer
echo "Stage 1: training unet vae..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=train_vae_unet_workspace \
    task=real_${TASK_NAME}_image_gelsight_emb_at_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=real_${TASK_NAME}_${TIMESTAMP} \
    at=vae_unet_wipe_lift \
    logging.mode=${LOGGING_MODE} \
    training.num_epochs=${num_epochs}


