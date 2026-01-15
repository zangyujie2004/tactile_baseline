from xarm.wrapper import XArmAPI
import time

if __name__ == '__main__':
    arm = XArmAPI('192.168.1.239')

    if arm.warn_code != 0:
        arm.clean_warn()
    if arm.error_code != 0:
        arm.clean_error()

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    speed = 0.6

    # angle = [1.478541, -0.524974, 0.076473, 1.297253, -0.071307, 1.764924, -1.59078]
    angle = [1.478278, -0.267359, -0.354714, 1.36086, -0.147519, 1.528497, -2.015091]
    arm.set_servo_angle(angle=angle, speed=speed, is_radian=True, wait=True)
    time.sleep(1)
    angle = [1.478288, -0.166976, -0.292118, 1.391008, -0.102355, 1.403859, -1.958061]
    arm.set_servo_angle(angle=angle, speed=speed, is_radian=True, wait=True)
    time.sleep(1)

    ### vase_sponge_test1
    # angle = [1.47827, -0.105039, -0.413837, 0.865564, -0.103457, 0.964442, -2.015087]
    # arm.set_servo_angle(angle=angle, speed=speed, is_radian=True, wait=True)
    # time.sleep(1)

    ### peel_cucumber
    # angle = [1.478748, -0.122841, -0.437593, 0.906287, -0.087462, 1.008328, -1.995694]
    # arm.set_servo_angle(angle=angle, speed=speed, is_radian=True, wait=True)
    # time.sleep(1)

    ### wipe_vase_ky
    # angle = [1.47827, -0.105039, -0.413837, 0.865564, -0.103457, 0.964442, -2.015087]
    # arm.set_servo_angle(angle=angle, speed=speed, is_radian=True, wait=True)
    # time.sleep(1)

    ### push_square
    angle = [1.478385, -0.280153, -0.391595, 0.949745, -0.157129, 1.18542, -1.99573]
    arm.set_servo_angle(angle=angle, speed=speed, is_radian=True, wait=True)
    time.sleep(1)


    arm.disconnect()