#!/usr/bin/env python3

"""
# 60hz: total time is 1.05s
python robot_arm/interpolate_control_position.py --pointnum 60
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

import argparse

def main(ip):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointnum", type=int, default=60, help="运动目标的个数, 为1代表只有终点和起点")
    parser.add_argument("--speed", type=int, default=120, help="速度")#注意，这个速度和角度的不一样，角度的0.6已经很快了

    args = parser.parse_args()
    pointnum = args.pointnum
    if pointnum < 60:
        pointnum = 60
    pointnum = 60
    speed = args.speed

    execute_hz = 60# 最低50
    execute_interval = 1.0 / execute_hz

    predict_execute_time = pointnum / execute_hz

    print(f"=========================================, pointnum is {pointnum}, speed is {speed}")
    print(f"predict_execute_time is {predict_execute_time}")

    servo = True

    start_pose = [207.446075, 371.847595, 331.218597, -178.274959, 2.825656, -179.190316] # example
    

    # end_pose = [315.270996, 371.847595, 599.053223, -178.274959, 2.825656, -179.190316] # 599.053223
    end_pose = [207.410767, 491.847595, 332.45575, -178.274959, 2.825656, -179.190316] # 599.053223
    # end_pose = [243.410767, 549.297974, 332.45575, 173.127907, 9.727849, 177.763479]


    arm = XArmAPI(ip)
    time.sleep(0.5)

    #clean error and warn
    if arm.warn_code != 0:
        arm.clean_warn()
    if arm.error_code != 0:
        arm.clean_error()

    arm.motion_enable(enable=True)

    arm.set_mode(0)
    arm.set_state(state=0)
    arm.set_position(x=start_pose[0], y=start_pose[1], z=start_pose[2], roll=start_pose[3], pitch=start_pose[4], yaw=start_pose[5], speed=speed, is_radian=False, wait=True)

    if not servo:
        arm.set_mode(0)
    else:
        arm.set_mode(1)

    arm.set_state(state=0)


    start = time.time()

    for point in range(1, pointnum+1):
        xyz = [0,0,0]
        for i in range(3):
            to = start_pose[i] + (end_pose[i] - start_pose[i])/pointnum * point
            xyz[i] = to
        # interpolate_pose = ...
        execute_start = time.time()
        if not servo:
            y=xyz[1]
            arm.set_position(x=xyz[0], y=y, z=xyz[2], roll=start_pose[3], pitch=start_pose[4], yaw=start_pose[5], speed=speed, is_radian=False, wait=False)
        else:
            x=xyz[0]
            if point % 2 ==1:
                # print("+")
                x += 3
            mv_position = [x, xyz[1], xyz[2], start_pose[3], start_pose[4], start_pose[5]]
            arm.set_servo_cartesian(mvpose=mv_position, speed=speed, is_radian=False)
            time.sleep(execute_interval)# execute_hz control
            print(f"real_time_speed is {arm.realtime_tcp_speed}")
        execute_end = time.time()
        print(f"execute time is {(execute_end - execute_start):.2f}")
        

    end = time.time()

    print(f"total time is {(end - start):.2f}\n\n")


    # arm.set_position(x=pose[0], y=pose[1], z=pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5], speed=1, is_radian=False, wait=True)

    # print(arm.get_position(is_radian=False))
    
    time.sleep(1.5)
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.set_position(x=start_pose[0], y=start_pose[1], z=start_pose[2], roll=start_pose[3], pitch=start_pose[4], yaw=start_pose[5], speed=speed, is_radian=False, wait=True)


if __name__ == '__main__':
    ip = '192.168.1.239'
    main(ip)