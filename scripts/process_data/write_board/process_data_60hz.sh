# end_frame多截取5frame，而vase_sponge_test1多截取20frame。因为vase_sponge_test1最后是抬高，而end_frame根据x轴来判断。所以vase_sponge_test1需要多截取
python scripts/process_data/write_board/process_data_all_zarr_60hz.py \
    --root_path /home/tars/projects/wky \
    --save_path /home/tars/projects/wky/processed_write_board/processed_60hz \
    --task_list write_board_0116_80 \
    --episode_length 80 
    # --save_camera_vis \
    # --episode_length 2

