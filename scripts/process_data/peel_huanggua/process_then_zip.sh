echo "process 15hz"
bash scripts/process_data/peel_huanggua/process_data.sh

echo "process 60hz"
bash scripts/process_data/peel_huanggua/process_data_60hz.sh

zip -r /media/tars/soon/processed_peel_huanggua.zip /media/tars/soon/processed_peel_huanggua

scp -P 1030 -r  /media/tars/soon/processed_peel_huanggua.zip root@8.130.212.67:/mnt/data/kywang/visual_tactile/dataset/