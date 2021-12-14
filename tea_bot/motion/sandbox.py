import scara as lib
from shapely import geometry
from matplotlib import pyplot as plt
import numpy as np

scara = lib.SCARA([[0,0],10,8])

o2 = [(5,-1),(5,1),(8,1),(8,-1)]
o3 = [(0,11),(5,15),(6,15),(6,13),(10,9),(9,8),(5,12)]

obs2 = geometry.Polygon(o2)
obs3 = geometry.Polygon(o3)

proj2 = scara.case2_proj_GN(obs2,'right')
proj3 = scara.case3_proj_GN(obs3)

scara.plot_poly(obs2,'r')
scara.plot_poly(obs3,'r')

scara.plot_poly(proj2,'g')
scara.plot_poly(proj3,'g')

ax,ay = scara.make_arc(0,0,scara.r1-scara.r2,[0,2*np.pi],200)
bx,by = scara.make_arc(0,0,scara.r1,[0,2*np.pi],200)
cx,cy = scara.make_arc(0,0,scara.r1+scara.r2,[0,2*np.pi],200)

# plt.plot(ax,ay)
# plt.plot(bx,by)
# plt.plot(cx,cy)

obstacle_list = [obs2,obs3,proj2,proj3]

path = scara.RRT([3,15],[7,-5],obstacle_list,15000,1)


#scara = lib.SCARA([[-86.25,-2.5],100,100])
#theta = scara.getMotorAngles(np.array([[-36.2,151.4],[-36.2,202],[138,137]]),'+')
