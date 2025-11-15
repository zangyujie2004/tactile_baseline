1. 在dataset_env中进行开环推理, 同时存储每个step的action chunk.
`bash scripts/eval_robot_from_dataset.sh`
其中，可以在`eval_robot_from_dataset.py`中修改存储位置以及推理的步数. 存储默认在`data/outputs/vis_outputs`

2. 可视化
`python tools/visualize_open_loop_xyz.py`
可以在可视化脚本里面调节可视化的维度以及存储位置，默认为`data/outputs/vis_outputs`

3. 打印对应step的image
`python tools/visualize_dataset.py`
将数据集中每一步的image存储。由于数据集没有被shuffle，因此可以和第一步和第二步的结果进行对照。存储默认在`data/outputs/vis_outputs`。

4. [TODO] 将2和3的结果结合在一起。