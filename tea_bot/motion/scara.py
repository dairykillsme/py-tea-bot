import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import csv
import pandas as pd
from shapely import geometry

'''
# TODO:
    add mode to getXY
    case 2 (generic/numeric)
    case 3 (generic/numeric)
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

            # if draw == True:
            #     '''
            #     ##########################
            #     ##########################
            #     THIS PART IS HORRIBLY SLOW
            #     AND JANKY
            #     ##########################
            #     ##########################
            #     '''
            #     trace_rev = xy_array[0:i]
            #     trace_fwd = trace_rev[::-1]
            #
            #     vertices = np.concatenate((vertices,
            #                                 trace_fwd,
            #                                 trace_rev))
            #
            # close = np.array([
            #     xy_array[i],
            #     B[i],
            #     self.A
            # ])
            #
            # vertices = np.concatenate((vertices, close))
            #
            # patch.set_xy(vertices)

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
        plt.show()

        return projection

    def case1_proj_GN(self,obstacle):
        '''
        numerically compute projection a case 1 obstacle

        input:
            obstacle: [shapely polygon object]
                obstacle to project

        output:
            projection: [shapely polygon object]
                projection of obstacle

        '''

        # theta = np.zeros([1,2])
        # theta[0,0] = np.pi/2
        # theta[0,1] = np.pi
        #
        # arm = self.angle2arm_poly(theta)
        #
        # plt.plot(*arm.xy)
        # plt.plot(*obstacle.exterior.xy)
        # plt.show()

        # get centroid of obstacle
        centroid = list(obstacle.centroid.coords)[0]
        # angle to centroid
        thetao = np.arctan2((centroid[1]-self.A[1]),(centroid[0]-self.A[0]))

        # make line string of arm thru centroid
        theta = np.zeros([1,2])
        theta[0,0] = thetao
        theta[0,1] = np.pi

        #sweep size
        sweep_deg = .1
        sweep_rad = sweep_deg/180*np.pi

        # sweep + to get alpha_plus
        intersect = True
        while intersect == True:
            theta[0,0] = theta[0,0] + sweep_rad
            arm = self.angle2arm_poly(theta)

            if not arm.intersects(obstacle):
                intersect = False
                alpha_plus = theta[0,0]
                self.plot_arm(arm)

        # sweep - to get alpha_minus
        theta[0,0] = thetao
        intersect = True
        while intersect == True:
            theta[0,0] = theta[0,0] - sweep_rad
            arm = self.angle2arm_poly(theta)

            if not arm.intersects(obstacle):
                intersect = False
                alpha_minus = theta[0,0]
                self.plot_arm(arm)

        ##################################
        # build projection from segments #
        ##################################
        n = 100 # number of points in each segment
        beta =  np.pi - alpha_plus

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
        x = np.concatenate((s1x,s2x,s3x,s4x))
        y = np.concatenate((s1y,s2y,s3y,s4y))

        # convert to list of ordered pairs
        points = list(zip(x,y))

        projection = geometry.Polygon(points)

        self.plot_poly(projection)
        self.plot_poly(obstacle)
        # plt.plot(self.A[0],self.A[1],'ro')
        plt.show()

        return projection,arm

    def case2_proj_GN(self,obstacle):


        return

    def make_arc(self,cx,cy,radius,angle_bounds,n):
        '''
        generate list of x and y points along a circular arc

        input:
            cx,yx   coord of center
            radius
            angle_bound     upper/lower bounds of theta
        '''
        angles = np.linspace(angle_bounds[0],angle_bounds[1],n)
        # angles = np.linspace(angle_lo,angle_hi,n)

        x = cx + radius*np.cos(angles)
        y = cy + radius*np.sin(angles)

        return x,y

    def plot_poly(self, poly):
        x,y = poly.exterior.xy                      # get x,y vertices
        plt.fill(x,y,alpha=.5,fc='r',ec='none')     # polygon fill
        plt.plot(*poly.exterior.xy,'r')             # polygon border

        return

    def plot_arm(self, arm):
        x,y = arm.xy
        plt.plot(x,y,'o',color='#999999')
        plt.plot(*arm.xy,'b')
        # plt.show()

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

if __name__ == "__main__":

    '''
    test cases:
    1   animate path
    2   animate path
    3   animate path
    4   animate path
    5   kinematics output against hand calcs
    6   case 1 circular obstacle projection
    7   forward kinematics
    8   case 1 generic / numeric
    '''

    test = 1

    if test == 1:
        sys = SCARA([[1,1],5,8])

        # numPts = 25
        # x = np.linspace(2,3,numPts)
        # y = np.linspace(2,5,numPts)
        # xy = np.zeros([numPts,2])
        # xy[:,0] = x
        # xy[:,1] = y

        xy = sys.xy_CSVToArray('controller.csv')

        theta = sys.getMotorAngles(xy, '+')

        sys.animatePath(theta, xy,
                        frameDelay=20,
                        width = 2,
                        save=False,
                        ghost=True,
                        draw=True)

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
        projection = scara.case1_proj_GN(box)

        points = [(1,1.75),(0.5,2),(-.25,1),(-1,1.1),(-1.5,.5),(1.5,1)]
        box = geometry.Polygon(points)
        projection = scara.case1_proj_GN(box)

        points = [(0,1),(0,2),(-.25,1)]
        box = geometry.Polygon(points)
        projection = scara.case1_proj_GN(box)

        points = [(0,1),(1,2),(-.5,2)]
        box = geometry.Polygon(points)
        projection = scara.case1_proj_GN(box)
