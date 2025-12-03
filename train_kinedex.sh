#!/bin/bash

CUDA_VISIBLE_DEVICES=1 accelerate launch train.py \
    --config-name=train_kinedex_image_workspace \
    task=kinedex \
    +task.dataset_path=data/ckpts/vase_sponge_test1/rdp_zarr \
    task.name=vase_sponge_test1_dp_kinedex \
    logging.mode=online







