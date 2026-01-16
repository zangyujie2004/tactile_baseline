echo "process 15hz"
bash scripts/process_data/write_board/process_data.sh

echo "process 60hz"
bash scripts/process_data/write_board/process_data_60hz.sh

scp -P 1030 -r  /home/tars/projects/wky/processed_write_board root@8.130.212.67:/mnt/data/kywang/visual_tactile/dataset/