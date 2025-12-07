import os
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
from multiprocessing import Process, set_start_method


def camera_worker(device_serial, device_name, feq, camera_id):
    """
    单独进程内采集并发布一个相机的彩色和深度图像
    """
    rospy.init_node(f'camera_node_{camera_id}', anonymous=True)
    image_pub = rospy.Publisher(f'/camera_{camera_id}_image', Image, queue_size=20)
    depth_pub = rospy.Publisher(f'/camera_{camera_id}_depth', Image, queue_size=20)
    bridge = CvBridge()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device_serial)
    if 'L515' in device_name:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    elif 'D405' in device_name or 'D435' in device_name:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    rate = rospy.Rate(feq)
    print(f'Start to publish camera_{camera_id} image and depth')
    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            current_time = rospy.Time.now()

            if not color_frame:
                continue
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32) 
            color_image = np.asanyarray(color_frame.get_data())
            
            color_msg = bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            color_msg.header.stamp = current_time
            color_msg.header.frame_id = f"camera_{camera_id}_frame"

            depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
            depth_msg.header.stamp = current_time
            depth_msg.header.frame_id = f"camera_{camera_id}_frame"
            
            image_pub.publish(color_msg)
            depth_pub.publish(depth_msg)
            rate.sleep()
    finally:
        pipeline.stop()
        print(f"Pipeline for {device_serial} stopped.")


class RealsenseMulti:
    def __init__(self, freq=30):
        self.freq = freq
        self.device_info = []
        for device in rs.context().devices:
            device_name = device.get_info(rs.camera_info.name)
            if device_name.lower() != 'platform camera':
                serial = device.get_info(rs.camera_info.serial_number)
                self.device_info.append((serial, device_name))
    
    def run(self):
        procs = []
        for i, (serial, name) in enumerate(self.device_info):
            camera_id = i+1
            p = Process(target=camera_worker, args=(serial, name, self.freq, camera_id))
            p.start()
            procs.append(p)
        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            for p in procs:
                p.terminate()
            print("All camera processes terminated.")


if __name__ == '__main__':
    set_start_method("spawn", force=True)  # 避免多进程兼容性问题
    realsense = RealsenseMulti(freq=15)
    realsense.run()