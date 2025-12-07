#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import threading
import time
from multiprocessing import Process, set_start_method


class CameraNode:
    def __init__(self, device_serial, device_name, camera_id, capture_freq=15, publish_freq=60):
        # 初始化 ROS 节点（每个进程只能调用一次！）
        rospy.init_node(f'camera_node_{camera_id}', anonymous=True)

        self.device_serial = device_serial
        self.device_name = device_name
        self.camera_id = camera_id
        self.capture_freq = capture_freq
        self.publish_freq = publish_freq

        # 共享 buffer（最新帧）
        self.latest_frame = {
            'color': None,
            'depth': None,
            'timestamp': rospy.Time(0)
        }
        self.buffer_lock = threading.Lock()

        # ROS 发布器
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher(f'/camera_{camera_id}/image', Image, queue_size=10)
        self.depth_pub = rospy.Publisher(f'/camera_{camera_id}/depth', Image, queue_size=10)

        # RealSense 配置
        self.align = rs.align(rs.stream.color)
        self.pipeline = None

    def start(self):
        """启动采集和发布线程，并保持主循环"""
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        capture_thread.start()
        publish_thread.start()

        # 主线程：等待 ROS shutdown
        try:
            while not rospy.is_shutdown():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    def _capture_loop(self):
        # 配置 RealSense
        config = rs.config()
        config.enable_device(self.device_serial)
        if 'L515' in self.device_name:
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        elif 'D405' in self.device_name or 'D435' in self.device_name:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            self.pipeline = rs.pipeline()
            self.pipeline.start(config)
            rospy.loginfo(f"Camera_{self.camera_id} (serial {self.device_serial}) capture started at {self.capture_freq} Hz")

            rate = rospy.Rate(self.capture_freq)
            while not rospy.is_shutdown():
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                ts = rospy.Time.now()

                with self.buffer_lock:
                    self.latest_frame['color'] = color_img.copy()
                    self.latest_frame['depth'] = depth_img.copy()
                    self.latest_frame['timestamp'] = ts

                rate.sleep()

        except Exception as e:
            rospy.logerr(f"Capture error in camera_{self.camera_id}: {e}")
        finally:
            if self.pipeline:
                self.pipeline.stop()
            rospy.loginfo(f"Camera_{self.camera_id} pipeline stopped.")

    def _publish_loop(self):
        rospy.loginfo(f"Camera_{self.camera_id} publish started at {self.publish_freq} Hz")
        rate = rospy.Rate(self.publish_freq)

        while not rospy.is_shutdown():
            with self.buffer_lock:
                color_img = self.latest_frame['color']
                depth_img = self.latest_frame['depth']
                ts = self.latest_frame['timestamp']

            if color_img is not None and depth_img is not None:
                try:
                    color_msg = self.bridge.cv2_to_imgmsg(color_img, encoding="bgr8")
                    color_msg.header.stamp = ts
                    color_msg.header.frame_id = f"camera_{self.camera_id}_link"
                    self.image_pub.publish(color_msg)

                    # depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
                    # depth_msg.header.stamp = ts
                    # depth_msg.header.frame_id = f"camera_{self.camera_id}_link"
                    # self.depth_pub.publish(depth_msg)
                except Exception as e:
                    rospy.logerr(f"Publish error in camera_{self.camera_id}: {e}")
            else:
                rospy.logwarn_throttle(5.0, f"Buffer empty for camera_{self.camera_id}")

            rate.sleep()


def run_camera_node(device_serial, device_name, camera_id, capture_freq, publish_freq):
    """供 multiprocessing.Process 调用的入口函数"""
    node = CameraNode(device_serial, device_name, camera_id, capture_freq, publish_freq)
    node.start()


class RealsenseMultiManager:
    def __init__(self, capture_freq=15, publish_freq=60):
        self.capture_freq = capture_freq
        self.publish_freq = publish_freq
        self.device_info = []

        ctx = rs.context()
        if len(ctx.devices) == 0:
            rospy.logerr("No RealSense devices found!")
            return

        for device in ctx.devices:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            if 'platform' not in name.lower():
                self.device_info.append((serial, name))
                rospy.loginfo(f"Found device: {name} (serial: {serial})")

        rospy.loginfo(f"Total usable RealSense devices: {len(self.device_info)}")

    def run(self):
        if not self.device_info:
            rospy.logwarn("No valid cameras to run.")
            return

        processes = []
        for i, (serial, name) in enumerate(self.device_info):
            camera_id = i + 1
            p = Process(
                target=run_camera_node,
                args=(serial, name, camera_id, self.capture_freq, self.publish_freq)
            )
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            rospy.loginfo("Terminating all camera processes...")
            for p in processes:
                p.terminate()
                p.join()
            rospy.loginfo("All processes terminated.")


if __name__ == '__main__':
    # 必须设置 spawn 模式以兼容 ROS + multiprocessing
    set_start_method("spawn", force=True)

    # 启动多相机管理器
    manager = RealsenseMultiManager(capture_freq=15, publish_freq=60)
    manager.run()