import scara as lib
from shapely import geometry
from matplotlib import pyplot as plt
import numpy as np

scara = lib.SCARA([[0,0],10,8])
theta = np.array([[100, 145]])
theta = theta/180*np.pi
scara.theta = theta
# scara.theta = theta

o1 = [(-1.5,0.5),(-1.5,1),(-.5,1),(-.5,.5)]
o2 = [(5,-1),(5,1),(8,1),(8,-1)]
o3 = [(4,12),(4,15),(6,15),(6,13),(10,9),(9,8),(5,12)]
o_oob = [(0,20),(5,20),(2.5,22.5)]

obs1 = geometry.Polygon(o1)
obs2 = geometry.Polygon(o2)
obs3 = geometry.Polygon(o3)
obs_oob = geometry.Polygon(o_oob)

obstacle_list = [obs1,obs2,obs3, obs_oob]

projection_list = scara.obstacle_projector(obstacle_list)

obs_and_proj = obstacle_list + projection_list

print('plotting obstacles and projections')
for poly in obs_and_proj:
    scara.plot_poly(poly,'r')

scara.auto_plot_arm()
scara.plot_workspace()

plt.show()
