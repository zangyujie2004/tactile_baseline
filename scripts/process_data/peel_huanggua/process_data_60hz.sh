# end_frame多截取5frame，而vase_sponge_test1多截取20frame。因为vase_sponge_test1最后是抬高，而end_frame根据x轴来判断。所以vase_sponge_test1需要多截取
python scripts/process_data/peel_huanggua/process_data_all_zarr_60hz.py \
    --root_path /media/tars/soon \
    --save_path /media/tars/soon/processed_peel_huanggua/peel_huanggua_60hz \
    --task_list peel_huanggua0107 \
    --episode_length 80
    # --save_camera_vis \
    # --episode_length 2

