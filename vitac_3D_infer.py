import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import pathlib
import sys
import os
import torch.nn.functional as F
from collections import deque
from scipy.spatial.transform import Rotation
from termcolor import cprint
from hydra.core.hydra_config import HydraConfig
from typing import Dict
from reactive_diffusion_policy.policy.diffusion_unet_3D_policy import DiffusionUnetImagePolicy
sys.path.append(os.path.dirname(__file__))
import copy
import collections
import dill
import threading
from reactive_diffusion_policy.common.pytorch_util import optimizer_to
from real_sensors import RealRobotEnv

# ======================== Main Inference Class ========================
input_key_list = ['left_robot_tcp_pose','left_gripper1_tactile', 'left_gripper2_tactile', 'global_pts']

def depth_image_to_camera_points(depth_image, color_image, intrinsic, mask=None):
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
    height, width = depth_image.shape 
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    Z = depth_image
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    point_cloud = np.dstack((X, Y, Z))
    if mask is not None:
        mask = (mask != 0).astype(np.bool).reshape(-1)
        pts = pts[mask]
    depth_mask = (depth_image > 0) * (depth_image < 1500)
    if color_image is not None:
        pts = np.concatenate((point_cloud, color_image), axis=-1)
        return pts.reshape(-1, 6), depth_mask
    else:
        return point_cloud.reshape(-1, 3), depth_mask

def signal_points_to_world(points, s2w):
    xyz = points[:, :3]
    xyz = np.dot(xyz, s2w[:3, :3].T) + s2w[:3, 3]
    if points.shape[1] == 6:
        rgb = points[:, 3:]
        return np.concatenate((xyz, rgb), axis=-1)
    else:
        return xyz

def pts_downsample(pointcloud, target_num, mode = 'uniform'):
    if mode == 'uniform':
        n = pointcloud.shape[0]
        if target_num >= n:
            return pointcloud.copy()
        idx = np.random.choice(n, target_num, replace=False)
        return pointcloud[idx]
    
CAM2_INTRINSIC = np.array([[905.998, 0, 651.417],
                           [0, 905.892, 360.766],
                           [0, 0, 1]])
CAM2_EXTRINSIC = np.array([[-0.667935, 0.509141, -0.517803, 612.403564],
                           [0.725919, 0.455896, -0.487969, 1034.735474],
                           [-0.013807, -0.709331, -0.682814, 1016.627197],
                           [0, 0, 0, 1]])


class RealWorld3DViTacInference:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)

        cprint(f"Loading checkpoint from: {cfg.inference.ckpt_path}", "yellow")
        payload = torch.load(cfg.inference.ckpt_path, map_location=self.device)
        train_cfg = payload['cfg']

        policy_cfg = train_cfg.policy
        self.env = RealRobotEnv(
                                n_obs_steps=policy_cfg.n_obs_steps,
                                robo_ip=cfg.inference.robot_ip)
        
        self.policy: DiffusionUnetImagePolicy = hydra.utils.instantiate(policy_cfg)
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.policy)
            self.policy = self.ema_model

        self.policy.load_state_dict(payload['state_dicts']['model'])
        
        self.policy.eval()
        self.policy.to(self.device)
        print("模型加载成功并已设为评估模式。")

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
            while True:
                obs = self.env.get_obs()
                if obs is None:
                    continue
                else:
                    if 'global_pts' in input_key_list:
                        for i in range(len(obs)):
                            depth = obs[i]['global_depth'] * 0.25
                            pts, depth_mask = depth_image_to_camera_points(depth, None, CAM2_INTRINSIC)
                            pts = signal_points_to_world(pts, CAM2_EXTRINSIC)
                            pts_mask = (pts[:, 1] > 300) * (pts[:, 1] < 800) * (pts[:, 0] > -440) * (pts[:, 0] < 350) * (pts[:, 2] > 75) * (pts[:, 2] < 550)
                            pts = pts[pts_mask * depth_mask.reshape(-1,)]
                            pts = pts_downsample(pts, 8192, mode='uniform')
                            obs[i]['global_pts'] = pts


                    obs_processed = {
                        key: torch.from_numpy(np.stack([o[key] for o in obs])).unsqueeze(0).to(self.device)
                        for key in obs[0].keys()
                    }

                    for key in input_key_list:
                        input_dict[key] = obs_processed[key]

                
                    # 使用模型进行动作预测
                    with torch.no_grad():
                        action_dict = self.policy.predict_action(input_dict)

                    # 提取动作序列
                    action_sequence = action_dict['action'].detach().cpu().numpy()[0]
                    
                    # 依次执行动作序列中的每个动作
                    for i in range(min(self.cfg.n_action_steps, len(action_sequence))):
                        action_step = action_sequence[i]
                        self.env.execute_action(action_step)
                    

        except KeyboardInterrupt:
            print("推理被用户中断。")
        finally:
            print("程序结束。")


@hydra.main(
    version_base=None,
    config_path="./reactive_diffusion_policy/config",
    config_name="real_wipe_image_3D_tactile"
)
def main(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    cfg.inference = {
        'ckpt_path': cfg.load_ckpt_path, # <--- 修改为你的模型路径
        'robot_ip': '192.168.1.239' # <--- 修改为你的机器人IP
    }
    OmegaConf.set_struct(cfg, True)
    inference_runner = RealWorld3DViTacInference(cfg)
    inference_runner.run()
    
if __name__ == "__main__":
    main()