import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from numpy.lib.function_base import angle

from .vision.map import WorldMap, CALIBRATION_FILE_DEFAULT
from .motion.scara import SCARA
#from .motion.stepper_control import MotionController

def theta_from_map_points(end_effector_point, joint_point):
    # determine if we are in + or -
    mode = ''
    angle_joint = np.arctan2(joint_point.y, joint_point.x)
    if angle_joint < 0:
        angle_joint += 2*np.pi
    angle_end = np.arctan2(end_effector_point.y, end_effector_point.x)
    if angle_end < 0:
        angle_end += 2*np.pi

    if (angle_end < angle_joint):
        mode = '+'
    else:
        mode = '-'
    
    return scara.getMotorAngles(np.array([[end_effector_point.x, end_effector_point.y]]), mode)

if __name__ == "__main__":
    matplotlib.use('tkagg') # need to use different backend
    scara = SCARA([[0,0], 9, 9])
    map = WorldMap(CALIBRATION_FILE_DEFAULT, show_feed=True)
    map.start()

    #  wait until ready and goal detected
    while not (map.goal_detected and map.ready):
        time.sleep(0.1)
    
    theta = theta_from_map_points(map.arm_end_effector, map.arm_joint)
    scara.theta = theta

    projection_list = scara.obstacle_projector(map.obstacle_list)
    obs_and_proj = map.obstacle_list + projection_list

    scara.auto_plot_arm()

    path = scara.RRT([map.arm_end_effector.x, map.arm_end_effector.y], [map.goal.x, map.goal.y], obs_and_proj, 5000, 1)

    map.stop()