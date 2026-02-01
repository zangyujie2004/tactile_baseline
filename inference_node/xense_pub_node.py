from xensesdk import Sensor
import argparse
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import rospy
import numpy as np
import time

class XenseSensor:
    def __init__(self, cfgs, feq):
        cfg1, cfg2 = cfgs
        self.tac1 = Sensor.create(cfg1)
        self.tac2 = Sensor.create(cfg2)
        self.feq = feq

        rospy.init_node(f'tactile_node', anonymous=True)
        self.tac1_pub = rospy.Publisher(f'/tac1_data', PointCloud2, queue_size=20)
        self.tac2_pub = rospy.Publisher(f'/tac2_data', PointCloud2, queue_size=20)

    def run(self):
        print('Begin to pub tactile data')
        rate = rospy.Rate(self.feq)
        while not rospy.is_shutdown():
            mesh3d_1, flow3d_1, timestamp_1 = self.tac1.selectSensorInfo(
                Sensor.OutputType.Mesh3D,
                Sensor.OutputType.Mesh3DFlow,
                Sensor.OutputType.TimeStamp
            )
            header1 = std_msgs.msg.Header()
            header1.stamp = rospy.Time.now()
            mesh3d_1 = mesh3d_1.reshape(-1, 3)
            flow3d_1 = flow3d_1.reshape(-1, 3)
            tac_data_1 = np.concatenate((mesh3d_1, flow3d_1), axis=1)
            
            mesh3d_2, flow3d_2, timestamp_2 = self.tac2.selectSensorInfo(
                Sensor.OutputType.Mesh3D,
                Sensor.OutputType.Mesh3DFlow,
                Sensor.OutputType.TimeStamp
            )
            header2 = std_msgs.msg.Header()
            header2.stamp = rospy.Time.now()
            mesh3d_2 = mesh3d_2.reshape(-1, 3)
            flow3d_2 = flow3d_2.reshape(-1, 3)
            tac_data_2 = np.concatenate((mesh3d_2, flow3d_2), axis=1)

            fields1 = [
                        PointField('x', 0,  PointField.FLOAT32, 1),
                        PointField('y', 4,  PointField.FLOAT32, 1),
                        PointField('z', 8,  PointField.FLOAT32, 1),
                        PointField('dx', 12, PointField.FLOAT32, 1),
                        PointField('dy', 16, PointField.FLOAT32, 1),
                        PointField('dz', 20, PointField.FLOAT32, 1),
                    ]
            cloud_msg1 = pc2.create_cloud(header1, fields1, tac_data_1)
            self.tac1_pub.publish(cloud_msg1)

            fields2 = [
                        PointField('x', 0,  PointField.FLOAT32, 1),
                        PointField('y', 4,  PointField.FLOAT32, 1),
                        PointField('z', 8,  PointField.FLOAT32, 1),
                        PointField('dx', 12, PointField.FLOAT32, 1),
                        PointField('dy', 16, PointField.FLOAT32, 1),
                        PointField('dz', 20, PointField.FLOAT32, 1),
                    ] 
            cloud_msg2 = pc2.create_cloud(header2, fields2, tac_data_2)
            self.tac2_pub.publish(cloud_msg2)   
            
            rate.sleep()

if __name__ == '__main__':
    # cfgs = ['OG000027', 'OG000222']
    # cfgs = ['OG000238', 'OG000236']# 可以改这样马
    # cfgs = ['OG000238', 'OG000561']
    cfgs = ['OG000238','OG000561']
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=int, default=30)
    args = parser.parse_args()
    tacxense = XenseSensor(cfgs, args.freq)
    tacxense.run()