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
from reactive_diffusion_policy.model.vision.pointnet2_utils.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import reactive_diffusion_policy.model.vision.pointnet2_utils.pointnet2.pytorch_utils as pt_utils
import copy
import collections
import dill
import threading
from reactive_diffusion_policy.common.pytorch_util import optimizer_to

# ======================== Placeholder Hardware Interfaces ========================
# 你需要根据你的硬件SDK替换这些伪代码
OmegaConf.register_new_resolver("eval", eval, replace=True)
class PlaceholderXArm:
    """占位符：代表你的xArm机器人控制器"""
    def __init__(self, ip_address):
        print(f"正在连接到位于 {ip_address} 的xArm...")
        self.ip = ip_address
        print("xArm 连接成功。")

    def set_pose(self, pose_mm_deg, speed=50, wait=True):
        """
        用给定的姿态移动机器人
        pose_mm_deg: [x, y, z, roll, pitch, yaw] (mm, degrees)
        """
        print(f"机器人移动到: {np.round(pose_mm_deg, 2)} (mm, deg), 速度: {speed}, 等待: {wait}")
        # 在这里替换为真实的机器人控制代码
        # self.arm.set_servo_cartesian(transform=[...], speed=speed, mvacc=2000, wait=wait)
        pass

    def get_tcp_pose(self):
        """获取机器人当前的TCP姿态，返回9D表示 [pos, 6D_rot]"""
        # 这里返回一个假的姿态数据，你需要从机器人SDK获取真实数据
        # 真实数据应为 [x, y, z, r11, r12, r13, r21, r22, r23]
        # 其中 r_ij 是旋转矩阵的元素
        print("正在获取TCP姿态...")
        # 伪造一个旋转矩阵
        rot_mat = Rotation.from_euler('xyz', [np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)]).as_matrix()
        # 伪造一个位置
        pos = np.array([0.4, 0.1, 0.2]) + np.random.uniform(-0.05, 0.05, 3)
        
        # 展平旋转矩阵并与位置拼接
        pose_9d = np.concatenate([pos, rot_mat.flatten()[:6]]) # Use first 6 elements of flattened matrix for 6D rot
        return pose_9d

class PlaceholderCamera:
    """占位符：代表你的相机"""
    def __init__(self):
        print("正在初始化相机...")
        print("相机准备就绪。")

    def get_point_cloud(self):
        """获取点云数据"""
        print("正在捕获点云...")
        # 返回一个假的随机点云，你需要从相机SDK获取真实数据
        return np.random.rand(8192, 3).astype(np.float32)

class PlaceholderTactileSensor:
    """占位符：代表你的触觉传感器"""
    def __init__(self, name):
        self.name = name
        print(f"正在初始化触觉传感器 {self.name}...")
        print(f"触觉传感器 {self.name} 准备就绪。")

    def get_tactile_point_cloud(self):
        """获取触觉点云数据 (位置+力)"""
        print(f"正在从 {self.name} 读取触觉数据...")
        # 返回一个假的随机触觉数据，你需要从传感器SDK获取真实数据
        # 形状应为 [700, 6] (xyz + 3D force/displacement)
        return np.random.rand(700, 6).astype(np.float32)

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
                


# ======================== Helper Functions ========================

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rotation_matrix_to_euler_angles(rotation_matrix, order='xyz', degrees=True):
    """将旋转矩阵转换为欧拉角"""
    r = Rotation.from_matrix(rotation_matrix)
    return r.as_euler(order, degrees=degrees)

# ======================== Main Inference Class ========================

class RealWorld3DViTacInference:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        
        # 1. 初始化设备
        self.device = torch.device(cfg.training.device)
        
        # 2. 加载策略模型
        self.policy = hydra.utils.instantiate(cfg.policy).to(self.device)
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.policy)
            self.policy = self.ema_model

        ckpt_path = pathlib.Path(cfg.inference.ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            
        print(f"正在从 {ckpt_path} 加载权重...")
        state_dict = torch.load(ckpt_path, map_location=self.device)
        
        if 'state_dicts' in state_dict:
            model_state_dict = state_dict['state_dicts']['model']
        else:
            model_state_dict = state_dict['model']
        
        self.policy.load_state_dict(model_state_dict)
        
        self.policy.eval()
        self.policy.to(self.device)
        print("模型加载成功并已设为评估模式。")

        # 3. 初始化硬件
        self.robot = PlaceholderXArm(ip_address=cfg.inference.robot_ip)
        self.camera = PlaceholderCamera()
        self.tactile1 = PlaceholderTactileSensor(name='gripper1')
        self.tactile2 = PlaceholderTactileSensor(name='gripper2')

        # 4. 初始化观测缓冲区
        self.obs_buffer = ObservationBuffer(maxlen=8)

    def get_obs(self) -> Dict[str, np.ndarray]:
        """从所有硬件获取一帧观测数据"""
        obs_processed = {
            'global_pts': self.camera.get_point_cloud(),
            'left_gripper1_tactile': self.tactile1.get_tactile_point_cloud(),
            'left_gripper2_tactile': self.tactile2.get_tactile_point_cloud(),
            'left_robot_tcp_pose': self.robot.get_tcp_pose()
        }

        return obs_processed
    
    
    
    def process_action(self, action: np.ndarray) -> np.ndarray:
        """处理策略输出的动作，转换为机器人可执行的格式"""
        if action.ndim != 1:
            raise ValueError(f"期望动作为1维, 但得到形状 {action.shape}")

        pos = action[:3]
        rot_6d = torch.from_numpy(action[3:9]).unsqueeze(0)
        
        rot_mat = rotation_6d_to_matrix(rot_6d).squeeze(0).numpy()

        euler_deg = rotation_matrix_to_euler_angles(rot_mat, degrees=True)

        pos_mm = pos * 1000

        executable_action = np.concatenate([pos_mm, euler_deg])
        return executable_action

    def run(self):
        """主推理循环"""
        print("Start inference loop...")

        try:
            while True:
                obs = self.get_obs()
                self.obs_buffer.append_obs(obs)
                obs_list = self.obs_buffer.get_new_obs(self.cfg.n_obs_steps)
                if obs_list is None:
                    continue
                else:
                    
                # 将多帧观测数据堆叠成一个批次
                    obs_processed = {
                        key: torch.from_numpy(np.stack([o[key] for o in obs_list])).unsqueeze(0).to(self.device)
                        for key in obs_list[0].keys()
                    }
                
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
    config_path=str(pathlib.Path(__file__).parent.joinpath("reactive_diffusion_policy/config")), 
    config_name="train_diffusion_unet_real_3dtactile_workspace"
)
def main(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    cfg.inference = {
        'ckpt_path': cfg.load_ckpt_path, # <--- 修改为你的模型路径
        'robot_ip': '192.168.1.209' # <--- 修改为你的机器人IP
    }
    OmegaConf.set_struct(cfg, True)

    inference_runner = RealWorld3DViTacInference(cfg)
    inference_runner.run()
    
if __name__ == "__main__":
    ROOT_DIR = str(pathlib.Path(__file__).parent)
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    main()