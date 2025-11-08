#!/usr/bin/env python3

"""
Description: record data (action and timestamp) and save as pkl
"""

import argparse
import time
import rospy
from geometry_msgs.msg import PoseStamped
from xarm.wrapper import XArmAPI

class Record_Xarm:
    def __init__(self, ip, feq):
        self.feq = feq
        self.arm = XArmAPI(ip)

        rospy.init_node('xarm_recorder', anonymous=True)
        self.eef_pub = rospy.Publisher('/xarm_eef_pose', PoseStamped, queue_size=20)
        time.sleep(0.5)

    def pub_data(self):
        print(f'start to pub trajectory with Feq={self.feq}')
        rate = rospy.Rate(self.feq)
        while not rospy.is_shutdown():
            eef_pose = self.arm.get_position(is_radian=False)[1]
            current_time = rospy.Time.now()
            pose_msg = PoseStamped()
            pose_msg.header.stamp = current_time
            pose_msg.header.frame_id = "base_link"
            pose_msg.pose.position.x = eef_pose[0]
            pose_msg.pose.position.y = eef_pose[1]
            pose_msg.pose.position.z = eef_pose[2]
            pose_msg.pose.orientation.x = eef_pose[3]
            pose_msg.pose.orientation.y = eef_pose[4]
            pose_msg.pose.orientation.z = eef_pose[5]
            pose_msg.pose.orientation.w = 1.0
            self.eef_pub.publish(pose_msg)
            
            rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ip',
        type=str ,
        default='192.168.1.239', 
        help='xArm IP address')
    parser.add_argument(
        "--feq", 
        type=int, 
        default=60, 
        help="Frequency of Pub"
    )
    args = parser.parse_args()

    xarm_recorder = Record_Xarm(args.ip, feq=60)
    xarm_recorder.pub_data()