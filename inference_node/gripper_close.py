import gripper

"""
close the gripper
"""

gripper = gripper.RobotiqGripper()
### vase_sponge_test1
# gripper.move(position=110, speed=255, force=1) # with tactile

### peel_cucumber
gripper.move(position=220, speed=255, force=1) # with tactile