# python tools/process_data_all.py \
#     --root_path /home/tars/projects/raw_data \
#     --save_path /home/tars/projects/dataset \
#     --task_list vase_sponge_test \
    # --save_camera_vis &

# python tools/process_data_all_zarr_60hz.py \
#     --root_path /home/tars/projects/raw_data \
#     --save_path /home/tars/projects/dataset_60z \
#     --task_list vase_sponge_test1 \
#     # --episode_length 2
#     # --save_camera_vis 


python tools/process_data_all_zarr_60hz.py \
    --root_path /media/tars/94fb725e-95de-474c-8a2b-1453309cff60/tactile_dataset \
    --save_path /home/tars/projects/dataset_60z \
    --task_list vase_sponge_test1 \
    --episode_length 2 \
    --save_camera_vis 


# python tools/process_data_all.py \
#     --root_path /home/tars/projects/raw_data \
#     --save_path /home/tars/projects/dataset \
#     --task_list vase_sponge_test vase_sponge_test1 


# python tools/process_data_all.py \
#     --root_path /mnt/sda/tactile_dataset/vase/ \
#     --save_path /home/tars/projects/dataset \
#     --task_list vase2_new_A vase2_new_B vase2_new_C 
    # --save_camera_vis


# python tools/process_data_all.py \
#     --root_path /mnt/sda/tactile_dataset/vase/ \
#     --save_path /home/tars/projects/dataset \
#     --task_list vase2_new_B &
#     # --save_camera_vis

# python tools/process_data_all.py \
#     --root_path /mnt/sda/tactile_dataset/vase/ \
#     --save_path /home/tars/projects/dataset \
#     --task_list vase2_new_C \
    # --save_camera_vis