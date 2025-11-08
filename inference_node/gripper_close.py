import gripper

"""
close the gripper
"""

gripper = gripper.RobotiqGripper()
gripper.move(position=110, speed=255, force=1) # with tactile
