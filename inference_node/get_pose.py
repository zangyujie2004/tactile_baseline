import time
import numpy as np
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R

class XArmController:
    def __init__(self, ip='192.168.1.228'):
        self.arm = XArmAPI(ip)
        time.sleep(0.5)
        self.clean_errors()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.tcp_transform = np.array([[0, 1, 0, 0],
                                       [0, 0, 1, 70.75],
                                       [1, 0, 0, 46],
                                       [0, 0, 0, 1]])
        # self.actions = np.load(action_file)[:128]

    def clean_errors(self):
        if self.arm.warn_code != 0:
            self.arm.clean_warn()
        if self.arm.error_code != 0:
            self.arm.clean_error()

    def move_to_pose(self, action):
        """
        parameters:
        input:
        set_position x,y,z unit in mm, roll,pitch,yaw unit in degree

        return: x,y,z unit in mm, roll,pitch,yaw unit in degree
        """
        # print(f"Executing action: {action}")
        # x,y,z unit in mm, roll,pitch,yaw unit in degree
        self.arm.set_position(x=action[0], y=action[1], z=action[2], roll=action[3], pitch=action[4], yaw=action[5], speed=100, is_radian=False, wait=True)
        
    def get_pose(self):
        eef_pose = self.arm.get_position(is_radian=False)[1]
        h2b = R.from_euler('xyz', eef_pose[3:], degrees=True).as_matrix()
        state = np.eye(4)
        state[:3, :3] = h2b
        state[:3, 3] = eef_pose[:3]
        transform_state = state @ self.tcp_transform
        euler = R.from_matrix(transform_state[:3, :3]).as_euler('xyz', degrees=True)
        translation = transform_state[:3, 3]
        eef_pose[3:] = euler
        eef_pose[:3] = translation
        joint_position = self.arm.get_servo_angle(is_radian=True)[1]
        print(joint_position)

        return eef_pose
    
if __name__ == "__main__":
    arm = XArmController()
    arm.get_pose()