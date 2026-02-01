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
from real_sensors_forcemimic import RealRobotEnv
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


input_key_list = ['global_pts', 'left_robot_tcp_pose', 'left_robot_gripper_width','left_gripper1_marker_offset_emb','left_gripper2_marker_offset_emb']

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

        try:
            rossub_thread = threading.Thread(target=self.env.ros_thread, daemon=True)
            rossub_thread.start()
            step_count = 0
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
                    
                    # 依次执行动作序列中的每个动作
                    for i in range(min(self.cfg.n_action_steps, len(action_sequence))):
                        action_step = action_sequence[i]
                        self.env.execute_action(action_step)
                    
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
 