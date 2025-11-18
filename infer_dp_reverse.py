"""
模型具有reverse的能力 \\
infer_dp.py和infer_dp_reverse.py目前都还没有relative的能力 \\
敲入回车, 机器就会停下来, 然后reverse执行
"""

import sys
import os
import pathlib
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
from termcolor import cprint
import copy
import threading
from real_sensors import RealRobotEnv
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

# 监控用户输入
import queue
import time
import select

def user_input_listener(input_queue):
    """后台线程，监听用户按回车"""
    while True:
        # 使用select监听是否有输入（非阻塞）
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            _ = sys.stdin.readline()  # 读取整行，但不使用内容
            if input_queue.empty():
                input_queue.put("ENTER")
        time.sleep(0.1)  # 避免占用CPU


input_key_list = ['left_wrist_img', 'left_robot_tcp_pose', 'left_robot_gripper_width']

class RealWorldDPInfer:
    def __init__(self, cfg: OmegaConf):
        # =========== Load configuration ===========
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # =========== Load checkpoint ===========
        cprint(f"Loading checkpoint from: {cfg.inference.ckpt_path}", "yellow")
        payload = torch.load(cfg.inference.ckpt_path, map_location=self.device)
        train_cfg = payload['cfg']
        
        # The policy configuration is saved within the checkpoint
        policy_cfg = train_cfg.policy
        self.env = RealRobotEnv(
                                n_obs_steps=policy_cfg.n_obs_steps,
                                pca_load_dir=cfg.inference.pca_path,
                                robo_ip=cfg.inference.robot_ip)
        
        # Instantiate the policy
        self.policy: DiffusionUnetImagePolicy = hydra.utils.instantiate(policy_cfg)
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.policy)
            self.policy = self.ema_model
        
        # Load the model weights
        self.policy.load_state_dict(payload['state_dicts']['model'])
        
        
        # Move policy to the correct device and set to evaluation mode
        self.policy.to(self.device)
        self.policy.eval()

        # =========== Initialize observation buffer ===========
        self.n_obs_steps = policy_cfg.n_obs_steps
        # Get the observation keys from the training config's shape_meta
        self.key_to_shape = train_cfg.shape_meta['obs']


    def run(self):
        """主推理循环"""
        print("Start inference loop...")
        input_dict = dict()


        input_queue = queue.Queue()  # 用于接收用户输入事件
        # 启动独立线程监听键盘输入
        listener_thread = threading.Thread(target=user_input_listener, args=(input_queue,), daemon=True)
        listener_thread.start()
        print("启动监听线程, 键入回车就可以让机器执行reverse")

        try:
            rossub_thread = threading.Thread(target=self.env.ros_thread, daemon=True)
            rossub_thread.start()
            step_count = 0
            should_reverse = False
            reverse_hoziron = self.policy.reverse_length # 往回走几步
            while True:
                obs = self.env.get_obs()
                if obs is None:
                    continue
                else:
                # 将多帧观测数据堆叠成一个批次
                    obs_processed = {
                        key: torch.from_numpy(np.stack([o[key] for o in obs])).unsqueeze(0).to(self.device)
                        for key in obs[0].keys()
                    }
                    # Data Processing
                    for key in obs_processed.keys():
                        if 'img' in key:
                            obs_processed[key] = obs_processed[key].permute(0, 1, 4, 2, 3)  # BNHWC -> BNCHW
                            obs_processed[key] = obs_processed[key].float() / 255.0
                    for key in input_key_list:
                        input_dict[key] = obs_processed[key]
                    
                    # Data Processing
                    # 使用模型进行动作预测
                    with torch.no_grad():
                        action_dict = self.policy.predict_action(input_dict)

                    # 提取动作序列
                    action_sequence = action_dict['action'].detach().cpu().numpy()[0]
                    action_reverse_sequence = action_dict['action_reverse'].detach().cpu().numpy()[0]
                    
                    # 依次执行动作序列中的每个动作
                    for i in range(min(self.cfg.n_action_steps, len(action_sequence))):
                        action_step = action_sequence[i]
                        if not input_queue.empty():
                            event = input_queue.get()
                            if event == "ENTER":
                                print("🚨 检测到用户按下回车, 进入 reverse 模式!先暂停两秒, 然后reverse执行")
                                should_reverse = True
                                time.sleep(2)  # 暂停2秒
                                break

                        self.env.execute_action(action_step)
                    
                    if should_reverse:
                        for i in range(min(reverse_hoziron, len(action_reverse_sequence))):
                            reverse_action_step = action_reverse_sequence[i]
                            self.env.execute_action(reverse_action_step)
                            print(f"往回走哟")
                        should_reverse = False
                        print(f"reverse执行完毕, 将继续正向执行")
                    
                    step_count += 1
                    if step_count >= self.env.max_steps:
                        print(f"已执行{50}步，推理循环结束。")
                        break  # 或者用 break 跳出 while True

        except KeyboardInterrupt:
            print("推理被用户中断。")
        finally:
            print("程序结束。")

@hydra.main(
    version_base=None,
    config_path="./reactive_diffusion_policy/config",
    config_name="dp_infer"
)
def main(cfg: OmegaConf):
    # Create the inference runner and start the loop
    OmegaConf.set_struct(cfg, False)
    cfg.inference = {
        'ckpt_path': cfg.load_ckpt_path,
        'pca_path': cfg.load_pca_path,
        'robot_ip': '192.168.1.239'
    }
    OmegaConf.set_struct(cfg, True)
    runner = RealWorldDPInfer(cfg)
    runner.run()

if __name__ == "__main__":
    main()
 