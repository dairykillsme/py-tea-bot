import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import csv
from numpy.lib.function_base import append
import pandas as pd
from shapely import geometry
from shapely.ops import nearest_points

'''
# TODO:
    RRT
        add smoothing
        initial check for start/goal in obstacle
        flag for when no path found
    fix case 1 narrow obstacle thing where arcs intersect
        look at if the two semicircles intersect
            if they do, then make a thin line between the two intersections
                this way you don't really lose any area of workspace BUT
                you still keep the path planner from sending the robot through the obstacle
        OR maybe split it into a left right case like case 2



    add mode to getXY
    case 2 (generic/numeric)
    look at unions
        https://shapely.readthedocs.io/en/stable/manual.html#efficient-unions
    test obstacle for what case it is
        if obstacle is multiple cases, split it
            use .difference() method
            need method to generate polygon of regions A,B,C
    deal with overlapping projection
        e.g. narrow case 1 obs
            scara = SCARA([[0,0],3,.5])
            points = [(0,1),(0,2),(-.25,1)]
            box = geometry.Polygon(points)
            projection = scara.case1_proj_GN(box)

    **** ADD HEURISTIC TO INVERSE KINEMATIC FOR MOTOR ANGLE 2.
        for mode +,
            theta2 = theta2
        for mode -,
            theta2 = 2*pi - theta2

    fix notation to match figures (d->e, p/m->e_x,e_y)
'''

class SCARA(object):
    def __init__(self,param):
        # A is [x,y] of motor 1
        # r1 is proximal arm length (motor to motor) (shoulder to elbow)
        # r2 is distal arm length (motor to EE) (elbow to hand)
        [self.A, self.r1, self.r2] = param
        self.theta = np.array([[0,0]])

    def getMotorAngles(self, xy_array, mode):
        '''
        inverse kinematic solver. get motor angles from xy points

        input:  numpy array of xy points
                mode = string (+,-)
        output: numpy array of motor angles
        '''

        xy = xy_array
        numPts = len(xy)

        #######################
        # initialze variables #
        #######################

        # empty d (apparent arm length)
        d_sqrd = np.zeros(numPts)

        # empt angles
        alpha = np.zeros(numPts)
        beta = np.zeros(numPts)
        theta2 = np.zeros(numPts)

        # empty motor angle >PAIRS<
        theta = np.zeros([numPts,2])

        # empty p and m (xy adjusted by location of motor1)
        p = np.zeros(numPts)
        m = np.zeros(numPts)

        ###############################
        # compute apparent arm and xy #
        ###############################

        p = xy[:,0]-self.A[0]
        m = xy[:,1]-self.A[1]

        d_sqrd = np.square(p) + np.square(m)

        ########################
        # compute motor angles #
        ########################

        #  alpha is angle from horiz to d
        alpha = np.arctan2(m,p)

        # beta is angle from d to r1
        beta = np.arccos((d_sqrd + self.r1**2 - self.r2**2)/(2*self.r1*np.sqrt(d_sqrd)))

        # theta 2 is motor 2 angle (elbow)
        theta2 = np.arccos((-d_sqrd + self.r1**2 + self.r2**2)/(2*self.r1*self.r2))

        if mode == '+':
            theta[:,0] = alpha + beta
            theta[:,1] = theta2
        elif mode == '-':
            theta[:,0] = alpha - beta
            theta[:,1] = 2*np.pi - theta2

        return theta #,alpha,beta,d_sqrd

    def getXY(self, motorAngles, mode):
        '''
        forward kinematic solver. get xy pairs from motor angle pairs
        '''
        numPts = len(motorAngles)

        # initialize elbow locs
        B = np.zeros([numPts,2])

        # initialize end effector locs
        C = np.zeros([numPts,2])

        # calculate x and y loc of elbow
        B[:,0] = self.A[0] + self.r1 * np.cos(motorAngles[:,0])
        B[:,1] = self.A[1] + self.r1 * np.sin(motorAngles[:,0])

        # calculate x and y of EE
        theta2_from_horizontal = motorAngles[:,1] + motorAngles[:,0] - np.pi # this is true for either mode!
        C[:,0] = B[:,0] + self.r2 * np.cos(theta2_from_horizontal)
        C[:,1] = B[:,1] + self.r2 * np.sin(theta2_from_horizontal)

        return B,C

    def animatePath(self, motorAngles, xy_array, frameDelay, width, save, draw, ghost):
        '''
        input:	motor angles and end effector points
        output:	animation of drawing

        args:
            width:  float   line width in plot
            save:   BOOL    true -> save mp4
            draw:   BOOL    true -> enable "pen" trace
            ghost:  BOOL    true -> show full path underlay
        '''

        numPts = len(motorAngles)

        # initialize elbow locs
        B = np.zeros([numPts,2])

        # calculate x and y loc of elbow
        B[:,0] = self.A[0] + self.r1 * np.cos(motorAngles[:,0])
        B[:,1] = self.A[1] + self.r1 * np.sin(motorAngles[:,0])

        fig = plt.figure()
        fig.set_dpi(100)
        fig.set_size_inches(7, 6.5)

        buffer = (self.r1 + self.r2)/2 * 0.1

        minx1 = np.min(xy_array[:,0])
        minx2 = np.min(B[:,0])
        minx  = np.min([minx1, minx2])
        maxx = np.max(xy_array[:,0])

        miny1 = np.min(xy_array[:,1])
        miny2 = np.min(B[:,1])
        miny  = np.min([miny1, miny2])
        maxy = np.max(xy_array[:,1])

        ax = plt.axes(xlim=(np.min([0,minx])-buffer, maxx+buffer),
                        ylim=(np.min([0, miny])-buffer, maxy+buffer))

        # vertices at t=0
        vertices = np.array([
            [self.A[0],self.A[1]],
            [B[0,0], B[0,1]],
            [xy_array[0,0], xy_array[0,1]]
        ])

        patch = patches.Polygon(vertices, edgecolor=[100/255,0,1], linewidth=width, closed=False, fill=False)

        def init():
            ax.add_line(patch)

            if ghost == True:
                # plot full path, which will be drawn on top of
                plt.plot(xy_array[:,0],xy_array[:,1])

            return patch,

        def animate(i):
            vertices = np.array([
                self.A,
                B[i],
                xy_array[i]
            ])

            trace = xy_array[0:i]
            trace = trace[::-1]

            vertices = np.concatenate((vertices,trace))

            patch.set_xy(vertices)

            fname = 'frame'+str(i)
            plt.savefig(fname)

            return patch,

        if draw == True:
            anim = animation.FuncAnimation(fig, animate,
                                            init_func = init,
                                            frames = numPts,
                                            interval = frameDelay,
                                            blit = True)
        else:
            plt.plot(xy_array[:,0],xy_array[:,1])

        if save == True:
            # Set up formatting for the movie files
            # Writer = animation.writers['ffmpeg']
            # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            # writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            print('tryna save...')
            f = r"D:\all my files\documents\uvm\5_masters\3_fall_21\autonomy\final_proj\code\py-tea-bot\tea_bot\motion"
            writergif = animation.PillowWriter(fps=30)
            anim.save(f, writer=writergif)

        plt.show()

        return

    def animatePath_but_better(self, motorAngles, xy_array, obstacle_list, frameDelay, width, save, draw, ghost):
        '''
        input:	motor angles and end effector points
        output:	animation of drawing

        args:
            width:  float   line width in plot
            save:   BOOL    true -> save mp4
            draw:   BOOL    true -> enable "pen" trace
            ghost:  BOOL    true -> show full path underlay
        '''

        numPts = len(motorAngles)

        # initialize elbow locs
        B = np.zeros([numPts,2])

        # calculate x and y loc of elbow
        B[:,0] = self.A[0] + self.r1 * np.cos(motorAngles[:,0])
        B[:,1] = self.A[1] + self.r1 * np.sin(motorAngles[:,0])

        fig = plt.figure()
        fig.set_dpi(100)
        fig.set_size_inches(7, 6.5)

        buffer = (self.r1 + self.r2)/2 * 0.1

        workspace_radius = (self.r1+self.r2)

        minx = self.A[0] - workspace_radius - buffer
        maxx = self.A[0] + workspace_radius + buffer

        miny = self.A[1] - workspace_radius - buffer
        maxy = self.A[0] + workspace_radius + buffer

        ax = plt.axes(xlim=(minx, maxx),
                        ylim=(miny, maxy))

        # vertices at t=0
        vertices = np.array([
            [self.A[0],self.A[1]],
            [B[0,0], B[0,1]],
            [xy_array[0,0], xy_array[0,1]]
        ])

        patch = patches.Polygon(vertices, edgecolor=[100/255,0,1], linewidth=width, closed=False, fill=False)

        def init():
            ax.add_line(patch)

            if ghost == True:
                # plot full path, which will be drawn on top of
                plt.plot(xy_array[:,0],xy_array[:,1])

            for obs in obstacle_list:
                self.plot_poly(obs,'r')
            self.plot_workspace()

            return patch,

        def animate(i):
            vertices = np.array([
                self.A,
                B[i],
                xy_array[i]
            ])

            trace = xy_array[0:i]
            trace = trace[::-1]

            vertices = np.concatenate((vertices,trace))

            patch.set_xy(vertices)

            return patch,

        if draw == True:
            anim = animation.FuncAnimation(fig, animate,
                                            init_func = init,
                                            frames = numPts,
                                            interval = frameDelay,
                                            blit = True)
        else:
            plt.plot(xy_array[:,0],xy_array[:,1])

        if save == True:
            # Set up formatting for the movie files
            # Writer = animation.writers['ffmpeg']
            # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            # writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            # anim.save("movie.mp4", writer=writer)
            p = 0

        plt.show()

        return

    def xy_CSVToArray(self, xy_CSV):
        '''
        Takes a csv file of xy points and converts to numpy array
        implemented with csv library
        '''
        ###########################################################
        # https://www.geeksforgeeks.org/working-csv-files-python/ #
        ###########################################################
        # csv file name
        # filename = "coordinator.csv"
        filename = xy_CSV

        # initializing the titles and rows list
        fields = []
        rows = []
        rawr =[]
        # reading csv file
        with open(filename, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
            # extracting field names through first row
            fields = next(csvreader)

            # extracting each data row one by one
            for row in csvreader:
                rows.append(row)

        # this array is made of strings so we need to convert
        str_array = np.asarray(rows)

        # cast string array into float array
        flt_array = str_array.astype(np.float)

        return flt_array

    def case1_proj_circularObs(self, ro, xo, yo):
        '''
        takes a circular, case 1 obstacle (within r1-r2) and projects it

        input:
            ro:     radius of obstacle
            xo, yo: x and y coord of center of obstacle
        '''

        [xobs, yobs] = self.make_arc(xo,yo,ro,[0,2*np.pi],100) # xy points of obstacle

        do = np.sqrt((self.A[0]-xo)**2+(self.A[1]-yo)**2) # center distance of obstacle

        thetao = np.arctan2(yo,xo) # angle to center of obstacle
        alpha = np.arcsin(ro/do) # angular width of obstacle
        beta = np.pi - alpha - thetao

        n = 100 # number of points for segments

        # segment 1
        cx1 = self.r1*np.cos(thetao + alpha) + self.A[0]
        cy1 = self.r1*np.sin(thetao + alpha) + self.A[1]
        angles1 = [-beta, np.pi-beta]
        radius1 = self.r2

        [s1x, s1y] = self.make_arc(cx1,cy1,radius1,angles1,n)

        # segment 2
        cx2 = self.A[0]
        cy2 = self.A[1]
        angles2 = [thetao+alpha, thetao-alpha]
        radius2 = self.r1 + self.r2

        [s2x, s2y] = self.make_arc(cx2,cy2,radius2,angles2,n)

        # segment 3
        cx3 = self.r1*np.cos(thetao - alpha) + self.A[0]
        cy3 = self.r1*np.sin(thetao - alpha) + self.A[1]
        angles3 =[thetao - alpha, thetao - alpha + np.pi]
        radius3 = self.r2

        [s3x, s3y] = self.make_arc(cx3,cy3,radius3,angles3,n)

        # segment 4
        cx4 = self.A[0]
        cy4 = self.A[1]
        angles4 = [thetao-alpha, thetao+alpha]
        radius4 = self.r1 - self.r2

        [s4x, s4y] = self.make_arc(cx4,cy4,radius4,angles4,n)

        # merge x and y
        x = np.concatenate((s1x,s2x,s3x,s4x))
        y = np.concatenate((s1y,s2y,s3y,s4y))

        # convert to list of ordered pairs
        points = list(zip(x,y))

        projection = geometry.Polygon(points)

        plt.plot(*projection.exterior.xy)
        plt.plot(xobs,yobs)
        plt.plot(self.A[0],self.A[1],'ro')
        # plt.show()

        return projection

    def case1_proj_GN(self,obstacle,obs_dir):
        '''
        numerically compute projection a case 1 obstacle

        input:
            obstacle: [shapely polygon object]
                obstacle to project
            obs_dir: 'left', 'right', 'both'

        output:
            projection: [shapely polygon object]
                projection of obstacle

        '''

        # get centroid of obstacle
        centroid = list(obstacle.centroid.coords)[0]
        # angle to centroid
        thetao = np.arctan2((centroid[1]-self.A[1]),(centroid[0]-self.A[0]))

        # arm extended straight thru centroid
        theta = np.zeros([1,2])
        theta[0,0] = thetao
        theta[0,1] = np.pi

        #sweep size
        sweep_deg = .1
        sweep_rad = sweep_deg/180*np.pi # in radians

        # sweep + to get alpha_plus
        intersect = True
        while intersect == True:
            theta[0,0] = theta[0,0] + sweep_rad
            arm = self.angle2arm_poly(theta)

            if not arm.intersects(obstacle):
                intersect = False
                alpha_plus = theta[0,0]
                # self.plot_arm(arm)

        # sweep - to get alpha_minus
        theta[0,0] = thetao
        intersect = True
        while intersect == True:
            theta[0,0] = theta[0,0] - sweep_rad
            arm = self.angle2arm_poly(theta)

            if not arm.intersects(obstacle):
                intersect = False
                alpha_minus = theta[0,0]
                # self.plot_arm(arm)

        ##################################
        # build projection from segments #
        ##################################
        n = 100 # number of points in each segment
        beta =  np.pi - alpha_plus

        # if doing projection directionally, need to buffer the edge
        buffer = 1/180*np.pi
        spread = np.arcsin(self.r2/self.r1) + buffer
        if obs_dir == 'left':
            alpha_plus = alpha_minus + spread
        elif obs_dir == 'right':
            alpha_minus = alpha_plus - spread

        # segment 1
        cx1 = self.r1*np.cos(alpha_plus) + self.A[0]
        cy1 = self.r1*np.sin(alpha_plus) + self.A[1]
        angles1 = [-beta, np.pi-beta]
        radius1 = self.r2

        [s1x, s1y] = self.make_arc(cx1,cy1,radius1,angles1,n)

        # segment 2
        cx2 = self.A[0]
        cy2 = self.A[1]
        angles2 = [alpha_plus, alpha_minus]
        radius2 = self.r1 + self.r2

        [s2x, s2y] = self.make_arc(cx2,cy2,radius2,angles2,n)

        # segment 3
        cx3 = self.r1*np.cos(alpha_minus) + self.A[0]
        cy3 = self.r1*np.sin(alpha_minus) + self.A[1]
        angles3 =[alpha_minus, alpha_minus + np.pi]
        radius3 = self.r2

        [s3x, s3y] = self.make_arc(cx3,cy3,radius3,angles3,n)

        # segment 4
        cx4 = self.A[0]
        cy4 = self.A[1]
        angles4 = [alpha_minus, alpha_plus]
        radius4 = self.r1 - self.r2

        [s4x, s4y] = self.make_arc(cx4,cy4,radius4,angles4,n)

        # merge x and y
        if obs_dir == 'both':
            x = np.concatenate((s1x,s2x,s3x,s4x))
            y = np.concatenate((s1y,s2y,s3y,s4y))
        elif obs_dir == 'left':
            x = np.concatenate((s2x,s3x,s4x))
            y = np.concatenate((s2y,s3y,s4y))
        elif obs_dir == 'right':
            x = np.concatenate((s1x,s2x,s4x))
            y = np.concatenate((s1y,s2y,s4y))

        # convert to list of ordered pairs
        points = list(zip(x,y))

        projection = geometry.Polygon(points)

        self.plot_poly(projection,'g')
        self.plot_poly(obstacle,'r')
        # plt.plot(self.A[0],self.A[1],'ro')
        # plt.show()

        return projection

    def case2_proj_GN(self,obstacle,obs_dir):
        # get centroid of obstacle
        centroid = list(obstacle.centroid.coords)[0]
        # angle to centroid
        thetao = np.arctan2((centroid[1]-self.A[1]),(centroid[0]-self.A[0]))

        # first link thru centroid, second link fully closed
        theta = np.zeros([1,2])
        theta[0,0] = thetao
        theta[0,1] = 0
        arm = self.angle2arm_poly(theta)

        #sweep size
        sweep_deg = .1
        sweep_rad = sweep_deg/180*np.pi # in radians

        if obs_dir == 'right':
            # first link thru centroid, second link fully closed
            theta[0,1] = 0
            arm = self.angle2arm_poly(theta)

            ##################
            # make segment 1 #
            ##################

            # sweep theta 1 away from centroid until no longer intersecting
            while arm.intersects(obstacle):
                theta[0,0] = theta[0,0] + sweep_rad
                arm = self.angle2arm_poly(theta)

            alpha_plus = theta[0,0]

            # need to get back into intersection
            while not arm.intersects(obstacle):
                theta[0,1] = theta[0,1] + sweep_rad
                arm = self.angle2arm_poly(theta)

            # now sweep thru obstacle until not intersecting
            while arm.intersects(obstacle):
                theta[0,1] = theta[0,1] + sweep_rad
                arm = self.angle2arm_poly(theta)

            # get intersection - to see why you need this, comment out
            # lines 531 and 532 where intersect is inserted into x,y:
            #   x = np.insert(x, 0, intersection[0])
            #   y = np.insert(y, 0, intersection[1])
            # and run following code:
            #   scara = lib.SCARA([[0,0],3,1])
            #   points = [(0,2.75),(.5,2.25),(.5,2.75)]
            #   obs = geometry.Polygon(points)
            #   projection = scara.case2_proj_GN(obs,'right')
            intersection = list(list(nearest_points(arm,obstacle)[0].coords)[0])

            B,C = self.getXY(theta,'+')
            B = B.tolist()[0]

            angle_lo = theta[0,0]+theta[0,1]-np.pi
            angle_hi = theta[0,0]
            x,y = self.make_arc(B[0],B[1],self.r2,[angle_lo, angle_hi],100)

            # append intersection to beginning of x,y List
            x = np.insert(x, 0, intersection[0])
            y = np.insert(y, 0, intersection[1])

            seg1 = list(zip(x,y))

            ######################
            # make easy segments #
            ######################
            angle_left = alpha_plus
            buffer = 5/180*np.pi
            angle_spread = np.arcsin(self.r2/self.r1) + buffer
            angle_right = alpha_plus - angle_spread

            x,y = self.make_arc(self.A[0],self.A[1],self.r1+self.r2,[angle_left,angle_right],100)
            seg2 = list(zip(x,y))

            x,y = self.make_arc(self.A[0],self.A[1],self.r1-self.r2,[angle_right,angle_left],100)
            seg3 = list(zip(x,y))

            ##################
            # make segment 4 #
            ##################
            theta[0,:] = [alpha_plus, 0]
            arm = self.angle2arm_poly(theta)

            seg4= []

            while theta[0,1] <= np.pi:
                while not arm.intersects(obstacle):
                    theta[0,1] = theta[0,1] + sweep_rad
                    arm = self.angle2arm_poly(theta)
                    if theta[0,1] > np.pi:
                        break
                B,C = self.getXY(theta,'+')
                seg4.append(C.tolist()[0])
                arm = self.angle2arm_poly(theta)

                while arm.intersects(obstacle):
                    theta[0,0] = theta[0,0] + sweep_rad
                    arm = self.angle2arm_poly(theta)

        elif obs_dir == 'left':
            # first link thru centroid, second link fully closed
            theta[0,1] = 2*np.pi
            arm = self.angle2arm_poly(theta)

            ##################
            # make segment 1 #
            ##################

            # sweep theta 1 away from centroid until no longer intersecting
            while arm.intersects(obstacle):
                theta[0,0] = theta[0,0] - sweep_rad
                arm = self.angle2arm_poly(theta)

            alpha_minus = theta[0,0]


            # need to get back into intersection
            while not arm.intersects(obstacle):
                theta[0,1] = theta[0,1] - sweep_rad
                arm = self.angle2arm_poly(theta)

            # now sweep thru obstacle until not intersecting
            while arm.intersects(obstacle):
                theta[0,1] = theta[0,1] - sweep_rad
                arm = self.angle2arm_poly(theta)

            # get intersection - to see why you need this, comment out
            # lines 618 and 619 where intersect is inserted into x,y:
            #   x = np.insert(x, 0, intersection[0])
            #   y = np.insert(y, 0, intersection[1])
            # and run following code:
            #   scara = lib.SCARA([[0,0],3,1])
            #   points = [(0,2.75),(.5,2.25),(.5,2.75)]
            #   obs = geometry.Polygon(points)
            #   projection = scara.case2_proj_GN(obs,'left')
            intersection = list(list(nearest_points(arm,obstacle)[0].coords)[0])

            B,C = self.getXY(theta,'-')
            B = B.tolist()[0]

            angle_lo = theta[0,0] + theta[0,1] - np.pi
            angle_hi = theta[0,0]
            x,y = self.make_arc(B[0],B[1],self.r2,[angle_lo, angle_hi],100)

            # append intersection to beginning of x,y List
            x = np.insert(x, 0, intersection[0])
            y = np.insert(y, 0, intersection[1])

            seg1 = list(zip(x,y))

            ######################
            # make easy segments #
            ######################
            angle_right = alpha_minus
            buffer = 5/180*np.pi
            angle_spread = np.arcsin(self.r2/self.r1) + buffer
            angle_left = alpha_minus + angle_spread

            x,y = self.make_arc(self.A[0],self.A[1],self.r1+self.r2,[angle_right,angle_left],100)
            seg2 = list(zip(x,y))

            x,y = self.make_arc(self.A[0],self.A[1],self.r1-self.r2,[angle_left,angle_right],100)
            seg3 = list(zip(x,y))

            ##################
            # make segment 4 #
            ##################
            theta[0,:] = [alpha_minus, 0]
            arm = self.angle2arm_poly(theta)

            seg4= []

            while theta[0,1] >= -np.pi:
                while not arm.intersects(obstacle):
                    theta[0,1] = theta[0,1] - sweep_rad
                    arm = self.angle2arm_poly(theta)
                    if theta[0,1] < -np.pi:
                        break
                B,C = self.getXY(theta,'+')
                seg4.append(C.tolist()[0])
                arm = self.angle2arm_poly(theta)

                while arm.intersects(obstacle):
                    theta[0,0] = theta[0,0] - sweep_rad
                    arm = self.angle2arm_poly(theta)

        seg4 = seg4[:-1]

        seg1 = np.asarray(seg1)
        seg2 = np.asarray(seg2)
        seg3 = np.asarray(seg3)
        seg4 = np.asarray(seg4)

        loop = np.concatenate((seg1,seg2,seg3,seg4))
        loop = np.asarray(loop)

        projection = geometry.Polygon(loop)

        self.plot_poly(obstacle,'r')
        self.plot_poly(projection,'g')
        # plt.show()

        return projection

    def case3_proj_GN(self,obstacle):
        # get centroid of obstacle as list of x,y
        centroid = list(list(obstacle.centroid.coords)[0])
        # angle to centroid
        thetao = np.arctan2((centroid[1]-self.A[1]),(centroid[0]-self.A[0]))

        # first link thru centroid, second link fully extended
        theta = np.zeros([1,2])
        theta[0,0] = thetao
        theta[0,1] = np.pi

        #sweep size
        sweep_deg = .1
        sweep_rad = sweep_deg/180*np.pi # in radians

        #############################
        # plus (+) side of obstacle #
        #############################
        seg1 = []
        arm = self.angle2arm_poly(theta)

        # while arm intersects, sweep arm out of obstacle
        # stopping once you DONT intersect
        while arm.intersects(obstacle):
            theta[0,0] = theta[0,0] + sweep_rad
            arm = self.angle2arm_poly(theta)

        # add to segment 1
        B,C = self.getXY(theta,'+')
        seg1.append(C.tolist()[0])

        # capture alpha_plus for later
        alpha_plus = theta[0,0]

        while theta[0,1] >=0: # stay in loop until arm has closed, at which point obstacle is cleared

            # while arm DOESNT intersect, sweep r2 toward obstacle
            # stop once arm intersects obstacle
            while not arm.intersects(obstacle):
                # increment theta2
                theta[0,1] = theta[0,1] - sweep_rad
                if theta[0,1] < 0:
                    # if arm is closed, break out of loop (obstacle cleared)
                    break
                arm = self.angle2arm_poly(theta)

            # add to segment 1
            B,C = self.getXY(theta,'+')
            seg1.append(C.tolist()[0])

            # update arm
            arm = self.angle2arm_poly(theta)

            # while arm instersects, sweep r1 away
            # stop once arm doesn't intersect
            while arm.intersects(obstacle):
                # increment theta1 and update arm
                theta[0,0] = theta[0,0] + sweep_rad
                arm = self.angle2arm_poly(theta)


        # algorithm will include final point as the arm fully closed and away from obstacle
        # we can just remove this value from our list
        seg1 = seg1[:-1]

        # once segment is complete, convert to numpy array
        seg1 = np.asarray(seg1)

        ##############################
        # minus (-) side of obstacle #
        ##############################
        seg3 = []

        theta[0,0] = thetao
        theta[0,1] = np.pi
        arm = self.angle2arm_poly(theta)

        # while arm intersects, sweep arm out of obstacle
        # stopping once you DONT intersect
        while arm.intersects(obstacle):
            theta[0,0] = theta[0,0] - sweep_rad
            arm = self.angle2arm_poly(theta)

        # add to segment 1
        B,C = self.getXY(theta,'-')
        seg3.append(C.tolist()[0])

        # capture alpha_minus for later
        alpha_minus = theta[0,0]

        while theta[0,1] <= 2*np.pi:
            # while arm DOESNT intersect, sweep arm toward obstacle
            # stop once arm intersects obstacle

            while not arm.intersects(obstacle):
                theta[0,1] = theta[0,1] + sweep_rad
                if theta[0,1] > 2*np.pi:
                    break
                arm = self.angle2arm_poly(theta)

            # add to segment 1
            B,C = self.getXY(theta,'+')
            seg3.append(C.tolist()[0])

            arm = self.angle2arm_poly(theta)

            # update theta1 and arm
            while arm.intersects(obstacle):
                theta[0,0] = theta[0,0] - sweep_rad
                arm = self.angle2arm_poly(theta)


        # algorithm will include final point as the arm fully closed and away from obstacle
        # we can just remove this value from our list
        seg3= seg3[:-1]

        # once segment is complete, convert to numpy array
        seg3 = np.asarray(seg3)

        # build arc from seg 1 to seg 3
        x,y = self.make_arc(self.A[0],self.A[1],self.r1+self.r2,[alpha_plus,alpha_minus],100)
        seg2 = list(zip(x,y))
        seg2 = np.asarray(seg2)

        #####################################################
        # put all segments into closed loop to make polygon #
        #####################################################

        # need to reverse order of seg1 so that its points are ordered
        #   clockwise around loop
        seg1 = np.flip(seg1,axis=0)

        # need to encode centroid as a numpy array point
        centroid_np = np.zeros([1,2])
        centroid_np[0] = centroid

        loop =  np.concatenate((seg1,seg2,seg3,centroid_np))
        loop = np.asarray(loop)

        projection = geometry.Polygon(loop)

        self.plot_poly(obstacle,'r')
        self.plot_poly(projection,'g')
        # plt.show()

        return projection

    def make_arc(self,cx,cy,radius,angle_bounds,n):
        '''
        generate list of x and y points along a circular arc

        input:
            cx,yx   coord of center
            radius
            angle_bound     upper/lower bounds of theta
        '''
        angles = np.linspace(angle_bounds[0],angle_bounds[1],n)

        x = cx + radius*np.cos(angles)
        y = cy + radius*np.sin(angles)

        return x,y

    def plot_poly(self, poly, color):
        x,y = poly.exterior.xy                      # get x,y vertices
        plt.fill(x,y,alpha=.3,fc=color,ec='none')     # polygon fill
        plt.plot(*poly.exterior.xy,color)             # polygon border

        return

    def plot_arm(self, arm):
        x,y = arm.xy
        plt.plot(x,y,'o',color='#999999')
        plt.plot(*arm.xy,'b')
        # plt.show()

        return

    def auto_plot_arm(self):
        arm = self.angle2arm_poly(self.theta)
        x,y = arm.xy
        plt.plot(x,y,'o',color='#999999')
        plt.plot(*arm.xy,'b')
        return

    def angle2arm_poly(self,angles):
        '''
        make a linestring object of arm from pair of angles
        '''
        [B,C] = self.getXY(angles,'mode')

        B = B[0].tolist()
        C = C[0].tolist()

        arm = geometry.LineString([self.A,B,C])

        # plt.plot(*arm.xy)
        # plt.show()

        return arm

    def plot_workspace(self):
        ax,ay = self.make_arc(0,0,self.r1-self.r2,[0,2*np.pi],200)
        bx,by = self.make_arc(0,0,self.r1,[0,2*np.pi],200)
        cx,cy = self.make_arc(0,0,self.r1+self.r2,[0,2*np.pi],200)

        plt.plot(ax,ay)
        plt.plot(bx,by)
        plt.plot(cx,cy)
        return

    def RRT(self, start, goal, obstacle_list, max_iteration, max_distance):
        # for simplicity, use rejection sampling:
        #   generate points in square that contains workspace
        #   if point is in workspace, continue with RRT
        #   if point not in workspace, generate new point
        # https://www.youtube.com/watch?v=4y_nmpv-9lI&ab_channel=nubDotDev
        #   since we are looking at annulus, different sampling method might
        #   be faster, but its not likely bc rejections sampling is easiest computation

        # generate possible x and y values
        n = 100
        x_vals = np.linspace(self.A[0]-self.r1-self.r2, self.A[0]+self.r1+self.r2, n)
        y_vals = np.linspace(self.A[1]-self.r1-self.r2, self.A[1]+self.r1+self.r2, n)

        # to keep points from projecting into the center region, make it an obstacle
        region_A = geometry.Point(self.A[0],self.A[1]).buffer(self.r1-self.r2)
        obstacle_list.append(region_A)

        ###############################
        # parameters and book keeping #
        ###############################
        goal_radius = 0.25
        num_nodes = 1 # how many nodes have been added to tree
        tree = np.zeros([max_iteration, 3])
        tree[0,:] = [start[0],start[1],0]
        # tree columns are:
        #     1: x coord of node
        #     2: y coord of node
        #     3: index of parent node

        # debug
        rndpts = np.zeros([max_iteration,3])

        for k in range(max_iteration):
            x = x_vals[np.random.randint(0,n)]
            y = y_vals[np.random.randint(0,n)]

            ########################
            # test if in workspace #
            ########################
            # radial dist to A
            r = ((x-self.A[0])**2 + (y-self.A[1])**2)**0.5
            # if r not within radii of annulus, then continue to next loop iteration
            if (r > self.r1+self.r2) or (r < self.r1-self.r2):
                continue

            # debug
            rndpts[k,0] = x
            rndpts[k,1] = y
            rndpts[k,2] = r

            #####################
            # find closest node #
            #####################
            all_dif = tree[0:num_nodes,0:2] - np.array([x,y])
            all_dist = np.sum(np.abs(all_dif)**2,axis=-1)**(1./2) # distances to all nodes
            close_index = np.argmin(all_dist) # index of closest node
            close_dist = all_dist[close_index] # distance of closest node

            ##################################
            # project node closer if too far #
            ##################################
            if close_dist > max_distance:
                vec = np.array([x,y]) - tree[close_index,0:2]           # vec from node to closest node on tree
                unit = vec / (np.sum(np.abs(vec)**2,axis=-1)**(1./2))   # unit vec from node to closest node on tree
                vec_prime = unit*max_distance                           # vec scaled by max dist
                x = tree[close_index,0] + vec_prime[0]                  # updated node, max dist away
                y = tree[close_index,1] + vec_prime[1]

            ########################################################
            # check new segment for intersection with any obstacle #
            ########################################################
            # segment between new candidate node and closest node
            segment = geometry.LineString([tree[close_index,0:2].tolist(), [x,y]])

            # initialize zeros for intersect flags
            intersect_flags = np.zeros(len(obstacle_list))

            # check intersection with every obstacle
            for j,obs in enumerate(obstacle_list):
                if obs.intersects(segment):
                    intersect_flags[j] = 1

            # if no intersections, then add node to tree
            if np.sum(intersect_flags) == 0:
                tree[num_nodes,0] = x
                tree[num_nodes,1] = y
                tree[num_nodes,2] = close_index
                num_nodes = num_nodes + 1

                goal_vec = np.array([x,y]) - np.array(goal)
                dist2goal = (np.sum(np.abs(goal_vec)**2,axis=-1)**(1./2))
                if dist2goal <= goal_radius:
                    print('path found at iteration',k)
                    break

            # flag for no path found
            if k == max(range(max_iteration))-1:
                print('max iteration reached. no path found')
                break

        ###########################
        # trace back optimal path #
        ###########################
        optimal_path = np.empty((0,2))
        parent = tree[num_nodes-1,:]

        while not parent[2] == 0:
            # while parent isn't first parent, append optimal path
            optimal_path = np.append(optimal_path,[parent[0:2]],axis=0)

            # get next parent
            parent = tree[int(parent[2]),:]

        # path is traced from finish back to start, so need to flip
        optimal_path = np.flip(optimal_path,axis=0)

        # path does not include start and goal, so append these
        optimal_path = np.vstack([start,optimal_path])
        optimal_path = np.vstack([optimal_path,goal])

        ax,ay = self.make_arc(0,0,self.r1-self.r2,[0,2*np.pi],200)
        bx,by = self.make_arc(0,0,self.r1,[0,2*np.pi],200)
        cx,cy = self.make_arc(0,0,self.r1+self.r2,[0,2*np.pi],200)

        # plt.plot(rndpts[:,0],rndpts[:,1],'b.',alpha=0.1)

        plt.plot(ax,ay)
        plt.plot(bx,by)
        plt.plot(cx,cy)

        plt.plot(optimal_path[:,0],optimal_path[:,1],'r',linewidth=3)
        plt.plot(start[0],start[1],'go')
        plt.plot(goal[0],goal[1],'ro')

        # plt.show()

        return optimal_path

    def path_smoother(self,path,obstacle_list,threshold,final_thresh):
        # print(path)
        # to keep points from projecting into the center region, make it an obstacle
        region_A = geometry.Point(self.A[0],self.A[1]).buffer(self.r1-self.r2)
        obstacle_list.append(region_A)

        ################################
        # split path into smaller bits #
        ################################
        n_pts = np.shape(path)[0]
        new_path=np.empty([1,2])
        cost=[]
        Ctmp=0

        for k in range(n_pts-1):
            p1=path[k,:]
            p2=path[k+1,:]
            AB=p2-p1
            ABunit=AB/np.linalg.norm(p1-p2)
            numBreaks=np.floor(np.linalg.norm(p1-p2)/threshold)

            for j in range(int(numBreaks)+1):
                inBetweenPnt=p1+ABunit*j*threshold
                new_path=np.append(new_path,[inBetweenPnt],axis=0)
                cost=np.append(cost,[Ctmp+j*threshold],axis=0)

            Ctmp=cost[-1]+np.linalg.norm(p2-new_path[-1,:])

        new_path=np.append(new_path,[p2],axis=0)
        cost=np.append(cost,[Ctmp],axis=0)
        new_path=np.delete(new_path,0,0)

        ##############################################################
        # pick 2 random points and see if connecting them is shorter #
        ##############################################################

        maxIter=200
        for i in range(maxIter):
            n_pts_new_path=np.shape(new_path)[0]
            # print('num points in new path ',n_pts_new_path)

            if n_pts_new_path<3:
                print('yo mama so fat, sh-')
                break

            rand_index1=np.random.randint(0,n_pts_new_path-1)
            rand_index2=np.random.randint(rand_index1+1,n_pts_new_path)
            rand_line= geometry.LineString([new_path[rand_index1,:].tolist(),new_path[rand_index2,:].tolist()])

            flag=False
            for obstacle in obstacle_list:
                if obstacle.intersects(rand_line):
                    flag= True
                    break
            if flag:
                continue
            cost_of_rand_line=cost[rand_index2]-cost[rand_index1]
            if not rand_line.length>=cost_of_rand_line:
                mommy = new_path[0:rand_index1+1,:]
                daddy = new_path[rand_index2:n_pts_new_path,:]
                new_path=np.concatenate((mommy,daddy),axis=0)

                mommy_bucks = cost[0:rand_index1+1]
                daddy_bucks = cost[rand_index2:n_pts_new_path]-cost_of_rand_line+rand_line.length
                cost=np.concatenate((mommy_bucks,daddy_bucks),axis=0)

        #######################################
        # split smooth path into smaller bits #
        #######################################
        # do this so that arm can follow short waypoints
        n_pts = np.shape(new_path)[0]
        smooth_path=np.empty([1,2])
        cost=[]
        Ctmp=0

        for k in range(n_pts-1):
            p1=new_path[k,:]
            p2=new_path[k+1,:]
            AB=p2-p1
            ABunit=AB/np.linalg.norm(p1-p2)
            numBreaks=np.floor(np.linalg.norm(p1-p2)/final_thresh)

            for j in range(int(numBreaks)+1):
                inBetweenPnt=p1+ABunit*j*final_thresh
                smooth_path=np.append(smooth_path,[inBetweenPnt],axis=0)
                cost=np.append(cost,[Ctmp+j*final_thresh],axis=0)

            Ctmp=cost[-1]+np.linalg.norm(p2-smooth_path[-1,:])

        smooth_path=np.append(smooth_path,[p2],axis=0)
        cost=np.append(cost,[Ctmp],axis=0)
        smooth_path=np.delete(smooth_path,0,0)

        #plottin' time
        plt.plot(smooth_path[:,0],smooth_path[:,1],linewidth=3)

        return smooth_path

    def obstacle_analyzer(self, obstacle_list):
        obs_dir_list = []
        obs_case_list = []

        for obs in obstacle_list:
            ##############################
            # get parameters of obstacle #
            ##############################
            # get centroid of obstacle
            centroid = list(obs.centroid.coords)[0]
            # angle to centroid
            thetao = np.arctan2((centroid[1]-self.A[1]),(centroid[0]-self.A[0]))
            # distance from centroid to first motor
            r = ((centroid[1]-self.A[1])**2 + (centroid[0]-self.A[0])**2)**0.5

            #############################
            # classify case of osbtacle #
            #############################
            # assuming obstacles do not cross regions....
            if r <= self.r1-self.r2:
                case = 1
            elif r <= self.r1:
                case = 2
            elif r <= self.r1+self.r2:
                case = 3
            else:
                case = 999
                print('obstacle out of range!')

            ##################################
            # classify direction of obstacle #
            ##################################
            # "unvwind" angles so they are bounded between 0 and 2pi
            theta1 = self.theta[0,0] % (2*np.pi)
            thetao = thetao % (2*np.pi)

            angle_dif = theta1 - thetao

            if angle_dif < -np.pi:
                dir = 'right'
            elif angle_dif < 0:
                dir = 'left'
            elif angle_dif > np.pi:
                dir = 'left'
            elif angle_dif > 0:
                dir = 'right'
            else:
                dir = 'UNKNOWN'
                print('uh oh cannot find direction of obstacle')

            ###################
            # append to lists #
            ###################
            obs_dir_list.append(dir)
            obs_case_list.append(case)

        return obs_dir_list, obs_case_list

    def obstacle_projector(self, obstacle_list):
        obs_dir_list, obs_case_list = self.obstacle_analyzer(obstacle_list)

        projection_list = []

        for obs, case, dir in zip(obstacle_list, obs_case_list, obs_dir_list):
            if case == 1:
                projection = self.case1_proj_GN(obs, dir)
            elif case == 2:
                projection = self.case2_proj_GN(obs, dir)
            elif case == 3:
                projection = self.case3_proj_GN(obs)
            else:
                continue
                print('problem with case class. no projection made')
            projection_list.append(projection)

        return projection_list

    def find_mode_give_path_angles(self, xy_path, obstacle_list):

        theta_pos_mode = self.getMotorAngles(xy_path,'+')
        theta_neg_mode = self.getMotorAngles(xy_path,'-')

        # assume positive mode. if + doesn't work, flip to -
        mode = '+'

        intersect_flag = False

        for angle_pair in theta_pos_mode:
            angle_pair = np.array([angle_pair])
            arm = self.angle2arm_poly(angle_pair)
            for obs in obstacle_list:
                if obs.intersects(arm):
                    intersect_flag = True
                    break
            if intersect_flag == True:
                mode = '-'
                break

        if mode == '+':
            motorAngles = theta_pos_mode
        else:
            motorAngles = theta_neg_mode

        return motorAngles

if __name__ == "__main__":

    '''
    test cases:
    1   animate path ** REQUIRES CONTROLLER.CSV
    2   animate path
    3   animate path
    4   animate path
    5   kinematics output against hand calcs
    6   case 1 circular obstacle projection (analytical)
    7   forward kinematics
    8   case 1 (generic/numeric)
    9   case 3 (generic/numeric)
    10  case 2 left and right (generic/numeric)
    11  all cases
    12  rrt
    13  one sided case 1
    14  obstacale analyzer and projector
    15  obstacale analyzer and projector w/ out of bounds obstacle
    16  find_mode_give_path_angles
    17  analyze obstacles, make rrt path, animate
    '''

    test = 17

    if test == 1:
        sys = SCARA([[1,1],5,8])

        # xy = sys.xy_CSVToArray('controller.csv')
        #
        # theta = sys.getMotorAngles(xy, '+')
        #
        # sys.animatePath(theta, xy,
        #                 frameDelay=20,
        #                 width = 2,
        #                 save=False,
        #                 ghost=True,
        #                 draw=True)

    elif test == 2:
        sys = SCARA([[0,0],1,1])

        xy = np.zeros([16,2])

        xy[:,0] = [1,-1,-1,1,
                    2,0,-2,0,
                    np.sqrt(1.99999), -np.sqrt(1.99999), -np.sqrt(1.99999), np.sqrt(1.99999),
                    np.sqrt(2), 0, -np.sqrt(2), 0]

        xy[:,1] = [1,1,-1,-1,
                    0,2,0,-2,
                    np.sqrt(1.99999), np.sqrt(1.99999), -np.sqrt(1.99999), -np.sqrt(1.99999),
                    0, np.sqrt(2), 0, -np.sqrt(2)]

        theta = sys.getMotorAngles(xy, '+')

        print(np.floor(theta*180/3.1415))

        sys.animatePath(theta, xy,
                        frameDelay=500,
                        width = 2,
                        save=False,
                        ghost=True,
                        draw=True)
    elif test == 3:
        sys = SCARA([[1,1],1,1])

        xy = np.zeros([16,2])

        xy[:,0] = [1,-1,-1,1,
                    2,0,-2,0,
                    np.sqrt(1.99999), -np.sqrt(1.99999), -np.sqrt(1.99999), np.sqrt(1.99999),
                    np.sqrt(2), 0, -np.sqrt(2), 0]

        xy[:,1] = [1,1,-1,-1,
                    0,2,0,-2,
                    np.sqrt(1.99999), np.sqrt(1.99999), -np.sqrt(1.99999), -np.sqrt(1.99999),
                    0, np.sqrt(2), 0, -np.sqrt(2)]

        xy = xy + 1

        theta = sys.getMotorAngles(xy, '+')

        print(np.floor(theta*180/3.1415))

        sys.animatePath(theta, xy,
                        frameDelay=500,
                        width = 2,
                        save=False,
                        ghost=True,
                        draw=True)
    elif test == 4:
        sys = SCARA([[0,0],2,3])

        xy = np.zeros([16,2])

        xy[:,0] = [1,-1,-1,1,
                    2,0,-2,0,
                    np.sqrt(1.99999), -np.sqrt(1.99999), -np.sqrt(1.99999), np.sqrt(1.99999),
                    np.sqrt(2), 0, -np.sqrt(2), 0]

        xy[:,1] = [1,1,-1,-1,
                    0,2,0,-2,
                    np.sqrt(1.99999), np.sqrt(1.99999), -np.sqrt(1.99999), -np.sqrt(1.99999),
                    0, np.sqrt(2), 0, -np.sqrt(2)]

        xy = xy*2

        theta = sys.getMotorAngles(xy, '+')

        print(np.floor(theta*180/3.1415))

        sys.animatePath(theta, xy,
                        frameDelay=2000,
                        width = 2,
                        save=False,
                        ghost=True,
                        draw=True)
    elif test == 5:
        sys = SCARA([[0,0],3,2])

        xy = np.zeros([4,2])
        xy[:,0] = [3,-2,-3,2]
        xy[:,1] = [2,3,-2,-3]

        theta = sys.getMotorAngles(xy,'+')

        print(np.floor(theta*180/3.1415))
    elif test == 6:
        # case1 circle projection test
        sys = SCARA([[0,0],2.5,.5])

        ro,xo,yo = [1,1,1]
        proj = sys.case1_proj_circularObs(ro, xo, yo)

        ro,xo,yo = [1,0,1]
        proj = sys.case1_proj_circularObs(ro, xo, yo)

        ro,xo,yo = [.5,-1,1]
        proj = sys.case1_proj_circularObs(ro, xo, yo)
    elif test == 7:
        sys = SCARA([[0,0],3,2])

        xy = np.zeros([4,2])
        xy[:,0] = [3,-2,-3,2]
        xy[:,1] = [2,3,-2,-3]

        theta = sys.getMotorAngles(xy,'+')

        [elbow, ee] = sys.getXY(theta,'mode')

        print(xy)
        print(ee)
    elif test == 8:
        scara = SCARA([[0,0],3,.5])

        points = [(.5,1.5),(-.5,1.5),(-.5,.5),(.5,.5)]
        box = geometry.Polygon(points)
        projection = scara.case1_proj_GN(box,'both')

        points = [(1,1.75),(0.5,2),(-.25,1),(-1,1.1),(-1.5,.5),(1.5,1)]
        box = geometry.Polygon(points)
        projection = scara.case1_proj_GN(box,'both')

        points = [(0,1),(0,2),(-.25,1)]
        box = geometry.Polygon(points)
        projection = scara.case1_proj_GN(box,'both')

        points = [(0,1),(1,2),(-.5,2)]
        box = geometry.Polygon(points)
        projection = scara.case1_proj_GN(box,'both')
    elif test == 9:
        scara = SCARA([[0,0],3,1])

        points = [(0,3.75),(.5,3.25),(.5,3.75)]
        obs = geometry.Polygon(points)
        projection = scara.case3_proj_GN(obs)

        points = [(0,3.25),(.5,3.25),(.5,3.75)]
        obs = geometry.Polygon(points)
        projection = scara.case3_proj_GN(obs)

        points = [(0,3.25),(.5,3.25),(.5,3.5),(.35,3.5),(.5,3.75)]
        obs = geometry.Polygon(points)
        projection = scara.case3_proj_GN(obs)
    elif test == 10:
        scara = SCARA([[0,0],3,1])

        points = [(0,2.75),(.5,2.25),(.5,2.75)]
        obs = geometry.Polygon(points)
        projection = scara.case2_proj_GN(obs,'right')

        points = [(0,2.75),(-.5,2.25),(-.5,2.75)]
        obs = geometry.Polygon(points)
        projection = scara.case2_proj_GN(obs,'left')
    elif test == 11:
        scara = SCARA([[0,0],10,8])

        o1 = [(-1.5,0.5),(-1.5,1),(-.5,1),(-.5,.5)]
        o2 = [(5,-1),(5,1),(8,1),(8,-1)]
        o3 = [(4,12),(4,15),(6,15),(6,13),(10,9),(9,8),(5,12)]

        obs1 = geometry.Polygon(o1)
        obs2 = geometry.Polygon(o2)
        obs3 = geometry.Polygon(o3)

        proj1 = scara.case1_proj_GN(obs1,'both')
        proj2 = scara.case2_proj_GN(obs2,'right')
        proj3 = scara.case3_proj_GN(obs3)

        scara.plot_poly(obs1,'r')
        scara.plot_poly(obs2,'r')
        scara.plot_poly(obs3,'r')

        scara.plot_poly(proj1,'g')
        scara.plot_poly(proj2,'g')
        scara.plot_poly(proj3,'g')

        ax,ay = scara.make_arc(0,0,scara.r1-scara.r2,[0,2*np.pi],200)
        bx,by = scara.make_arc(0,0,scara.r1,[0,2*np.pi],200)
        cx,cy = scara.make_arc(0,0,scara.r1+scara.r2,[0,2*np.pi],200)

        plt.plot(ax,ay)
        plt.plot(bx,by)
        plt.plot(cx,cy)

        plt.show()
    elif test == 12:
        scara = SCARA([[0,0],10,8])

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

        path = scara.RRT([3,15],[0,-5],obstacle_list,5000,1)
        # plt.plot(path[:,0],path[:,1])
        path1 = scara.path_smoother(path,obstacle_list,.5)
        path2 = scara.path_smoother(path1,obstacle_list,.5)
        path3 = scara.path_smoother(path2,obstacle_list,.5)
        print(path3)
        plt.show()

    elif test == 13:
        scara = SCARA([[0,0],10,8])

        o1 = [(-1.5,0.5),(-1.5,1),(-.5,1),(-.5,.5)]

        obs1 = geometry.Polygon(o1)

        proj1_L = scara.case1_proj_GN(obs1,'left')
        proj1_R = scara.case1_proj_GN(obs1,'right')
        proj1_B = scara.case1_proj_GN(obs1,'both')
    elif test == 14:
        scara = SCARA([[0,0],10,8])

        theta = np.array([[100, 145]])
        theta = theta/180*np.pi
        scara.theta = theta
        # scara.theta = theta

        o1 = [(-1.5,0.5),(-1.5,1),(-.5,1),(-.5,.5)]
        o2 = [(5,-1),(5,1),(8,1),(8,-1)]
        o3 = [(4,12),(4,15),(6,15),(6,13),(10,9),(9,8),(5,12)]

        obs1 = geometry.Polygon(o1)
        obs2 = geometry.Polygon(o2)
        obs3 = geometry.Polygon(o3)

        obstacle_list = [obs1,obs2,obs3]

        projection_list = scara.obstacle_projector(obstacle_list)

        obs_and_proj = obstacle_list + projection_list

        print('plotting obstacles and projections')
        for poly in obs_and_proj:
            scara.plot_poly(poly,'r')

        scara.auto_plot_arm()
        scara.plot_workspace()

        plt.show()
    elif test == 15:
        scara = SCARA([[0,0],10,8])

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

        obstacle_list = [obs1,obs2,obs3,obs_oob]

        projection_list = scara.obstacle_projector(obstacle_list)

        obs_and_proj = obstacle_list + projection_list

        print('plotting obstacles and projections')
        for poly in obs_and_proj:
            scara.plot_poly(poly,'r')

        scara.auto_plot_arm()
        scara.plot_workspace()

        plt.show()
    elif test == 16:
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
        mode_path_angles = scara.find_mode_give_path_angles(path, obstacle_list)
        scara.animatePath(mode_path_angles, path,
                                frameDelay=500,
                                width = 2,
                                save=False,
                                ghost=True,
                                draw=True)
    elif test == 17:

        scara = SCARA([[0,0],10,8])

        theta = np.array([[100, 145]])
        theta = theta/180*np.pi
        scara.theta = theta

        o1 = [(-1.5,0.5),(-1.5,1),(-.5,1),(-.5,.5)]
        o2 = [(5,-1),(5,1),(8,1),(8,-1)]
        o3 = [(4,12),(4,15),(6,15),(6,13),(10,9),(9,8),(5,12)]
        o_oob = [(0,20),(5,20),(2.5,22.5)]

        obs1 = geometry.Polygon(o1)
        obs2 = geometry.Polygon(o2)
        obs3 = geometry.Polygon(o3)

        plt.show()

        obstacle_list = [obs1,obs2,obs3]

        projection_list = scara.obstacle_projector(obstacle_list)


        obs_and_proj = obstacle_list + projection_list

        # print('plotting obstacles and projections')
        # for poly in obs_and_proj:
        #     scara.plot_poly(poly,'r')

        path = scara.RRT([3,15],[9,-5],obs_and_proj,20000,2)
        path = scara.path_smoother(path,obs_and_proj,1,.9)
        path = scara.path_smoother(path,obs_and_proj,.8,.1)

        plt.show()

        angles = scara.getMotorAngles(path, '+')

        scara.animatePath_but_better(angles, path, obs_and_proj,
                        frameDelay=1,
                        width = 2,
                        save=True,
                        ghost=True,
                        draw=True)
