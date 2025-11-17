"""
Temporal Ensemble.
"""

import torch
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R, Slerp
from reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix
import time
import threading

class AlignLazyController:
    """
    Alignment版本的EnsembleBuffer. 使用这个buffer的时候, 注意不要提前根据自定义的`latency_step`来进行截断 \\
    现在是实验版本, 即lazy地实现alignment, 被推理进程和执行进程调用, 而不是调用推理逻辑和执行逻辑 \\
    通过write_lock防止model和executor同时对action队列进行操作. 但目前只用了写锁, 依然会存在executor的读和model的写的冲突问题, 但我目前不理会 \\
    TODO: 实现ActiveAlignEnsembleBuffer \\
    Lazy版本的不方便实现`Batchstep`, 因为slow系统得到的是latent_action_chunk, 没有办法直接比较action之间的距离, 只能比较latent之间的相似度

    """
    def __init__(self,
                 ensemble_mode = "new",
                 execute_horizon = 5,
                 n_obs_steps = 2,
                 obs_temporal_downsample_ratio = 2,
                ):
        assert ensemble_mode in ["new", "old", "avg", "act", "hato"], f"Ensemble mode {ensemble_mode} not supported now."
        self.mode = ensemble_mode
        self.timestep = 0
        self.last_update_timestep = -execute_horizon
        self.latest_obs = None
        self.actions = []
        self.action_shape = None

        self.execute_hoziron = execute_horizon
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio

        self.write_lock = threading.Lock()

    def clear(self):
        """
        Clear the ensemble buffer.
        """
        self.timestep = 0
        self.latest_obs = None
        self.actions = []
        self.last_update_timestep = -self.execute_hoziron
    

    def need_update(self):
        return self.timestep - self.last_update_timestep >= self.execute_hoziron
    
    def get_latest_obs_timestep(self):
        """
        if last_obs is None, please get the obs manually!!!
        """
        return (self.latest_obs, self.timestep)

    def add_action(self, action_chunk, inf_timestep):
        """
        Add action to the ensemble buffer:

        Parameters:
        - action_chunk: horizon x action_dim (...);
        - inf_timestep: inference_timestep
        """
        if self.timestep - self.last_update_timestep < self.execute_hoziron:
            print(f"It's not the time to add_action")
            return
        
        with self.write_lock:
            action_chunk = np.array(action_chunk)
            if self.action_shape == None:
                self.action_shape = action_chunk.shape[1:]
                assert len(self.action_shape) == 1, "Only support action with 1D shape."
            else:
                assert self.action_shape == action_chunk.shape[1:], "Incompatible action shape."

            start = self.timestep - inf_timestep# 从开始推理开始, 执行了多少个step
            horizon = action_chunk.shape[0]

            self.actions = []
            print("============================== add action ======================================")
            # time.sleep(3)
            # horizon = start + self.execute_horizon # 用于debug，在慢系统sleep(2)的时候，让快系统也不要跑
            for i in range(start, horizon):
                # print(f"action_chunk[{i}] is {action_chunk[i][0:3]}")
                if len(self.actions) > i-start:
                    self.actions[i-start] = action_chunk[i]
                else:
                    self.actions.append(action_chunk[i])
            
            self.last_update_timestep = self.timestep
                
            print(f"向队列之中添加新的action, horizon is {action_chunk.shape[0]}, timestep is {self.timestep}, inf_timestep is {inf_timestep}, len(self.actions) is {len(self.actions)} start is {start}")
        
    def get_action(self):
        """
        Get ensembled action from buffer. \\
        执行完action之后, 记得update
        """
        if len(self.actions) == 0:
            return None
        
        action = self.actions[0]
        # action = self.actions.pop(0)
        # self.timestep += 1
        # 检查一下，是不是每次get的都是一个action（最新的那个），直到新添加队列
        print(f"[get action] timestep is {self.timestep} action[0:10] is {action[0:3]}")
        return action
    
    def update(self, env):
        """
        执行完毕, 更新latest_obs
        """
        with self.write_lock:# 使用lock 防止线程之间写冲突, 从而规避未定义行为
            print("update obs")
            obs = env.get_obs(
                        obs_steps=self.n_obs_steps,
                        temporal_downsample_ratio=self.obs_temporal_downsample_ratio)
            if len(self.actions) != 0:
                self.actions.pop(0)
            self.timestep += 1
            self.latest_obs = obs



