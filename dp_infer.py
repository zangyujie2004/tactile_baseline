import sys
import os
import pathlib
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
from collections import deque
import time
import collections
from termcolor import cprint
from typing import Dict
import copy
import threading
# Add the project root to the Python path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.common.precise_sleep import precise_sleep



class ObservationBuffer:

    def __init__(self, maxlen: int = 8):
        self._buf = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append_obs(self, obs):
        with self._lock:
            self._buf.append(obs)

    def get_new_obs(self, n_obs_steps):
        with self._lock:
            if len(self._buf) < n_obs_steps:
                return None
            else:
                obs_list = list(self._buf)[-n_obs_steps:]
                return obs_list

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

        self.obs_buffer = ObservationBuffer(maxlen=8)
        # =========== Initialize observation buffer ===========
        self.n_obs_steps = policy_cfg.n_obs_steps
        # Get the observation keys from the training config's shape_meta
        self.key_to_shape = train_cfg.shape_meta['obs']

    def get_obs(self) -> Dict[str, np.ndarray]:
        """从所有硬件获取一帧观测数据"""
        obs_processed = {
            'left_wrist_img': np.ones((240, 320, 3)),
            'left_robot_tcp_pose': np.ones((9)),
            'left_robot_gripper_width': np.ones((1)),
            'left_gripper1_marker_offset_emb': np.ones((15))
        }

        return obs_processed

    def process_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Post-process the raw action from the policy to a format the robot can execute.
        This might involve scaling, clipping, or converting coordinate frames.
        """
        # The env_runner has a method to process actions
        action = self.raw_action

        return action

    def run(self):
        """主推理循环"""
        print("Start inference loop...")

        try:
            while True:
                obs = self.get_obs()
                self.obs_buffer.append_obs(obs)
                obs_list = self.obs_buffer.get_new_obs(self.n_obs_steps)
                if obs_list is None:
                    continue
                else:
                    
                # 将多帧观测数据堆叠成一个批次
                    obs_processed = {
                        key: torch.from_numpy(np.stack([o[key] for o in obs_list])).unsqueeze(0).to(self.device)
                        for key in obs_list[0].keys()
                    }

                    for key in obs_processed.keys():
                        if 'img' in key:
                            obs_processed[key] = obs_processed[key].permute(0, 1, 4, 2, 3)  # BNHWC -> BNCHW
                            obs_processed[key] = obs_processed[key].float() / 255.0
                
                    # 使用模型进行动作预测
                    with torch.no_grad():
                        action_dict = self.policy.predict_action(obs_processed)

                    # 提取动作序列
                    action_sequence = action_dict['action'].detach().cpu().numpy()[0]
                    
                    # 依次执行动作序列中的每个动作
                    for i in range(min(self.cfg.n_action_steps, len(action_sequence))):
                        action_to_execute = action_sequence[i]
                        
                        # 确保动作形状正确
                        if action_to_execute.shape != (self.cfg.policy.shape_meta.action.shape[0],):
                            action_to_execute = action_to_execute.flatten()

                        # 将策略输出的动作转换为机器人可执行的格式
                        executable_action = self.process_action(action_to_execute)
                        
                        # 控制机器人执行动作
                        self.robot.set_pose(executable_action, wait=True)

                    print("-" * 20)
                    
                    # 获取新的观测并更新缓冲区
                    new_obs = self.get_obs()
                    self.obs_buffer.append(new_obs)

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
        'ckpt_path': cfg.load_ckpt_path, # <--- 修改为你的模型路径
        'robot_ip': '192.168.1.209' # <--- 修改为你的机器人IP
    }
    OmegaConf.set_struct(cfg, True)
    runner = RealWorldDPInfer(cfg)
    runner.run()

if __name__ == "__main__":
    main()
 