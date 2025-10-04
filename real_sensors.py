import serial
import time
import binascii
import threading
import torch
import requests
import numpy as np
from omegaconf import DictConfig
from xarm.wrapper import XArmAPI
from typing import Union, List, Dict, Optional
from loguru import logger
import os
import cv2
import rospy
import pickle
import collections
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, PointStamped
import sensor_msgs.point_cloud2 as pc2
import message_filters
from cv_bridge import CvBridge
import transforms3d as t3d



def pose_6d_to_4x4matrix(pose: np.ndarray) -> np.ndarray:
    # convert 6D pose (x, y, z, r, p, y) to 4x4 transformation matrix
    mat = np.eye(4)
    quat = t3d.euler.euler2quat(pose[3], pose[4], pose[5])
    mat[:3, :3] = t3d.quaternions.quat2mat(quat)
    mat[:3, 3] = pose[:3]
    return mat

def pose_6d_to_pose_9d(pose: np.ndarray) -> np.ndarray:
    """
    Convert 6D state to 9D state
    :param pose: np.ndarray (6,), (x, y, z, rx, ry, rz)
    :return: np.ndarray (9,), (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
    """
    rot_6d = pose_6d_to_4x4matrix(pose)[:3, :2].T.flatten()
    return np.concatenate((pose[:3], rot_6d), axis=0)

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector (batch * 3)
    """
    v_mag = np.linalg.norm(v, axis=1, keepdims=True)  # batch * 1
    v_mag = np.maximum(v_mag, 1e-8)
    v = v / v_mag
    return v

def ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix from ortho6d representation
    """
    x_raw = ortho6d[:, 0:3]  # batch * 3
    y_raw = ortho6d[:, 3:6]  # batch * 3
    x = normalize_vector(x_raw)  # batch * 3
    z = np.cross(x, y_raw)  # batch * 3
    z = normalize_vector(z)  # batch * 3
    y = np.cross(z, x)  # batch * 3

    x = x[:, :, np.newaxis]
    y = y[:, :, np.newaxis]
    z = z[:, :, np.newaxis]

    matrix = np.concatenate((x, y, z), axis=2)  # batch * 3 * 3
    return matrix


class ObservationBuffer:

    def __init__(self, maxlen: int = 8):
        self._buf = collections.deque(maxlen=maxlen)

    def append_obs(self, obs):
        self._buf.append(obs)

    def get_new_obs(self, n_obs_steps):
        if len(self._buf) < n_obs_steps:
            return None
        else:
            obs_list = list(self._buf)[-n_obs_steps:]
            return obs_list


class RobotiqGripper:
    def __init__(self, port='/dev/gripper_port', baudrate=115200, timeout=1):
        """
        初始化串口连接，设置默认通信参数。
        """
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
    
    def send_command(self, command):
        """
        发送指令到夹爪，并读取返回数据
        """
        self.ser.write(command)
        time.sleep(0.05)
        response = self.ser.read_all()
        return response

    def receive(self, command):
        """
        接收夹爪返回的数据
        """
        self.ser.write(command)
        time.sleep(0.05)
        response = self.ser.read_all()
        return response

    def activate_gripper(self):
        """
        激活夹爪
        """
        command = b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30'
        response = self.send_command(command)
        print(f"Activate Response: {binascii.hexlify(response)}")
        return response

    def deactivate_gripper(self):
        """
        复位夹爪
        """
        command = b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30'
        response = self.send_command(command)
        print(f"Deactivate Response: {binascii.hexlify(response)}")
        return response

    def close_gripper(self):
        """
        关闭夹爪
        """
        command = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29'
        response = self.send_command(command)
        # print(f"Close Gripper Response: {binascii.hexlify(response)}")
        return response

    def open_gripper(self):
        """
        打开夹爪
        """
        command = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19'
        response = self.send_command(command)
        # print(f"Open Gripper Response: {binascii.hexlify(response)}")
        return response

    def move(self, position, speed=255, force=255):
        """
        控制夹爪移动到任意指定位置（非阻塞优化）
        """
        if not (0 <= position <= 255):
            raise ValueError("目标位置必须在 0 到 255 的范围内！")
        if not (0 <= speed <= 255 or 0 <= force <= 255):
            raise ValueError("速度或力量值超出范围！")

        # 构造指令
        command = (
            b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00' +
            bytes([position, speed, force])
        )
        crc = self._calculate_crc(command)
        command += crc

        # 异步发送命令
        self.ser.write(command)

    def _calculate_crc(self, data):
        """
        计算 CRC 校验
        :param data: 待校验的字节数据
        :return: CRC16 校验码，返回低字节在前，高字节在后
        """
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc.to_bytes(2, byteorder='little')

    def get_gripper_status(self):
        """
        获取夹爪状态信息
        返回包含以下信息的字典:
        - gripper_status: 夹爪状态 (gACT, gGTO, gSTA)
        - object_status: 物体检测状态 (gOBJ)
        - fault_status: 故障状态 (gFLT)
        - position_request_echo: 位置请求回显 (gPR)
        - position: 当前位置 (gPO)
        - current: 当前电流 (gCU)
        """
        # 读取输入寄存器 (FC04) - 地址0x07D0开始，读取3个寄存器(6字节)
        command = b'\x09\x04\x07\xD0\x00\x03\xB1\xCE'
        # 读取响应 (11字节: 地址1 + 功能码1 + 字节数1 + 数据6 + CRC2)
        response = self.receive(command)

        if len(response) != 11:
            print("Error: Invalid response length")
            return None
        
        # 解析响应数据
        data = response[3:-2]  # 去掉地址、功能码、字节数和CRC
        # 按照文档中的寄存器映射解析数据
        status = {
            'gripper_status': {
                'gACT': (data[0] >> 0) & 0x01,  # 激活状态
                'gGTO': (data[0] >> 3) & 0x01,  # 动作状态
                'gSTA': (data[0] >> 4) & 0x03,  # 夹爪状态
                'gOBJ': (data[0] >> 6) & 0x03   # 物体检测状态
            },
            'fault_status': data[2],             # 故障状态
            'position_request_echo': data[3],    # 位置请求回显
            'position': data[4],                 # 当前位置
            'current': data[5]                   # 当前电流 (值×10 ≈ mA)
        }
        return status
    
    def get_gripper_extended_status(self):
        """
        获取详细的夹爪状态信息（包括人类可读的描述）
        """
        status = self.get_gripper_status()
        if status is None:
            return None
        
        # 详细状态描述
        gSTA_desc = {
            0x00: "Gripper is in reset (or automatic release) state",
            0x01: "Activation in progress",
            0x03: "Activation is completed"
        }
        gOBJ_desc = {
            0x00: "Fingers are in motion towards requested position. No object detected",
            0x01: "Fingers have stopped due to a contact while opening before requested position. Object detected opening",
            0x02: "Fingers have stopped due to a contact while closing before requested position. Object detected closing",
            0x03: "Fingers are at requested position. No object detected or object has been lost/dropped"
        }
        # 故障状态描述
        fault_desc = {
            0x00: "No fault (solid blue LED)",
            0x05: "Action delayed, the activation must be completed prior to performing the action",
            0x07: "The activation bit must be set prior to performing the action",
            0x08: "Maximum operating temperature exceeded",
            0x09: "No communication during at least 1 second",
            0x0A: "Under minimum operating voltage",
            0x0B: "Automatic release in progress",
            0x0C: "Internal fault",
            0x0D: "Activation fault",
            0x0E: "Overcurrent triggered",
            0x0F: "Automatic release completed"
        }
        
        # 添加描述信息
        gripper_status = status['gripper_status']
        gripper_status['gSTA_desc'] = gSTA_desc.get(gripper_status['gSTA'], "Unknown")
        gripper_status['gOBJ_desc'] = gOBJ_desc.get(gripper_status['gOBJ'], "Unknown")
        status['fault_desc'] = fault_desc.get(status['fault_status'], "Unknown fault")
        
        # 电流转换为实际值 (mA) 和计算扭矩
        status['current_mA'] = status['current'] * 10
        torque_constant = 0.02  # 假设电机力矩常数为 0.02 N·m / A
        status['motor_torque_Nm'] = (status['current_mA'] / 1000) * torque_constant
        
        return status

    def disconnect(self):
        """
        关闭串口连接
        """
        self.ser.close()


class XArmController:
    def __init__(self, ip='192.168.1.239'):
        self.arm = XArmAPI(ip)
        time.sleep(0.5)
        self.clean_errors()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
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
        pose = self.arm.get_position()[1]

        return pose


class RealRobotEnv:
    def __init__(self,
                 n_obs_steps: int = 2,
                 pca_load_dir: str = "/home/robotics/Prometheus/reactive_diffusion_policy/tactile_pca/vase",
                 robo_ip='192.168.1.239'
                 ):
        self.gripper = RobotiqGripper()
        self.motor = XArmController(ip=robo_ip)
        rospy.init_node('all_topics_subscriber', anonymous=True)
        self.obs_buffer = ObservationBuffer(maxlen=10)
        self.motor = XArmController()
        self.max_steps = 100
        self.left_tac_transform_matrix = np.load(os.path.join(pca_load_dir, 'pca_matrix1.npy'))
        self.left_tac_mean_matrix = np.load(os.path.join(pca_load_dir, 'pca_mean1.npy'))

        self.mutex = threading.Lock()
        self.bridge = CvBridge()
        self.n_obs_steps = n_obs_steps


    def ros_thread(self):
        self.sub_camera_1_image = message_filters.Subscriber('/camera_1_image', Image)
        self.sub_camera_2_image = message_filters.Subscriber('/camera_2_image', Image)
        self.sub_camera_1_depth = message_filters.Subscriber('/camera_1_depth', Image)
        self.sub_camera_2_depth = message_filters.Subscriber('/camera_2_depth', Image)
        self.sub_tac1_data = message_filters.Subscriber('/tac1_data', PointCloud2)
        self.sub_tac2_data = message_filters.Subscriber('/tac2_data', PointCloud2)
        self.sub_xarm_eef = message_filters.Subscriber('/xarm_eef_pose', PoseStamped)
        self.sub_gripper_pos = message_filters.Subscriber('/gripper_tele_pos', PointStamped)  
        self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.sub_camera_1_image,
                 self.sub_camera_2_image,
                 self.sub_camera_1_depth,
                 self.sub_camera_2_depth,
                 self.sub_tac1_data,
                 self.sub_tac2_data,
                 self.sub_xarm_eef,
                 ],
                queue_size=100,
                slop=0.8,
                allow_headerless=False
            )
        self.ts.registerCallback(self.synced_callback)
        rospy.spin()

    def process_tac_data_msg(self, tac_data):
        points = []
        for p in pc2.read_points(tac_data, field_names=("x", "y", "z", "dx", "dy", "dz"), skip_nans=True):
            points.append(p)
        tac_data_array = np.array(points)
        mesh = tac_data_array[:,:3]
        deform = tac_data_array[:,3:]
        
        return mesh, deform

    def synced_callback(self, camera_1_image, camera_2_image, camera_1_depth, camera_2_depth, tac1_data, tac2_data, xarm_eef):
        raw_obs_dict = {}
        mesh1, deform1 = self.process_tac_data_msg(tac1_data)
        mesh2, deform2 = self.process_tac_data_msg(tac2_data)
        
        deform_emb = (deform1[:,:-1].reshape(-1, 1)[0] - self.left_tac_mean_matrix) @ self.left_tac_transform_matrix

        cam1_img = self.bridge.imgmsg_to_cv2(camera_1_image, desired_encoding='bgr8')

        cam2_depth = self.bridge.imgmsg_to_cv2(camera_2_depth, desired_encoding='32FC1')

        tx = xarm_eef.pose.position.x,
        ty = xarm_eef.pose.position.y,
        tz = xarm_eef.pose.position.z,
        rr = xarm_eef.pose.orientation.x,
        rp = xarm_eef.pose.orientation.y,
        ry = xarm_eef.pose.orientation.z,
        xarm_eef_array = np.array([tx, ty, tz, rr, rp, ry])[:, 0]
        xarm_eef_array[:3] *= 0.001
        xarm_eef_array[3:] = np.radians(xarm_eef_array[3:])
        xarm_eef_array = pose_6d_to_pose_9d(xarm_eef_array)
        
        gripper_pos_array = 112
        
        raw_obs_dict['left_robot_tcp_pose'] = xarm_eef_array
        raw_obs_dict['left_robot_gripper_width'] = np.array([gripper_pos_array / 255.0])
        raw_obs_dict['left_wrist_img'] = cv2.resize(cam1_img, (320, 240))
        raw_obs_dict['left_gripper1_marker_offset_emb'] = deform_emb
        raw_obs_dict['left_gripper1_tactile'] = np.concatenate((mesh1, deform1), axis=-1)
        raw_obs_dict['left_gripper2_tactile'] = np.concatenate((mesh2, deform2), axis=-1)
        raw_obs_dict['global_depth'] = cam2_depth

        self.obs_buffer.append_obs(raw_obs_dict)

    def get_obs(self) -> Dict[str, np.ndarray]:
        
        obs = self.obs_buffer.get_new_obs(self.n_obs_steps)
        if obs is None:
            return None

        return obs
    
    def post_process_action(self, action: np.ndarray) -> np.ndarray:
        left_rot_mat_batch = ortho6d_to_rotation_matrix(action[None, 3:9])  # (action_steps, 3, 3)
        left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
        left_trans_batch = action[None, :3]  # (action_steps, 3)
        left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)

        return left_action_6d[0]


    def execute_action(self, action: np.ndarray) -> None:
        action = self.post_process_action(action)
        action[:3] *= 1000.0
        action[3:6] = np.rad2deg(action[3:6])
        left_tcp_target_6d_in_robot = action[:6]
        logger.debug(f"robot_action: {left_tcp_target_6d_in_robot}")
        self.motor.move_to_pose(left_tcp_target_6d_in_robot)   
