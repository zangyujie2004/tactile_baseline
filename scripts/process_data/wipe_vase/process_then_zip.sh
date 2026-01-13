echo "process 15hz"
bash scripts/process_data/wipe_vase/process_data.sh

echo "process 60hz"
bash scripts/process_data/wipe_vase/process_data_60hz.sh

scp -P 1030 -r  /home/tars/projects/wky/processed_wipe_vase root@8.130.212.67:/mnt/data/kywang/visual_tactile/dataset/