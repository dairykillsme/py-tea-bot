import scara as lib
from shapely import geometry
from matplotlib import pyplot as plt
import numpy as np

scara = lib.SCARA([[0,0],10,8])

path = np.array([[4,15],[6,13],[7,12],[8,11]])

theta_p = scara.getMotorAngles(path,'+')
theta_m = scara.getMotorAngles(path,'-')

'''
scara.animatePath(theta_p, path,
                        frameDelay=500,
                        width = 2,
                        save=False,
                        ghost=True,
                        draw=True)

scara.animatePath(theta_m, path,
                        frameDelay=500,
                        width = 2,
                        save=False,
                        ghost=True,
                        draw=True)
'''

obstacle_list = []
mode_path = scara.find_mode_give_path(path, obstacle_list)
scara.animatePath(mode_path, path,
                        frameDelay=500,
                        width = 2,
                        save=False,
                        ghost=True,
                        draw=True)
