from math import fmod

# Let this run as a test on computer
try:
    from adafruit_motorkit import MotorKit
    RASPI = True
except NotImplementedError:
    RASPI = False

import time
import numpy as np

class MotionController:
    THETA_1_RADS_STEP = np.pi / 100
    THETA_2_RADS_STEP = np.pi / 200
    
    def __init__(self, theta1_initial, theta2_inital) -> None:
        if RASPI:
            self.kit = MotorKit()
        self.theta1_target = 0
        self.theta2_target = 0
        if (theta1_initial < 0):
            self.theta1_real = theta1_initial + 2*np.pi
        else:
            self.theta1_real = theta1_initial
        self.theta2_real = theta2_inital
    
    def set_theta1(self, theta1):
        if (theta1 < 0):
            self.theta1_target = theta1 + 2*np.pi
        else:
            self.theta1_target = theta1
    
    def set_theta2(self, theta2):
        self.theta2_target = theta2
    
    def shortest(self, dest, source):
        distance = fmod((dest - source + 2*np.pi), 2*np.pi)
        return distance, distance < np.pi
    
    def tick(self):
        theta1_err, theta1_dir = self.shortest(self.theta1_target, self.theta1_real)
        if (np.abs(theta1_err) > MotionController.THETA_1_RADS_STEP):
            if theta1_dir:
                if RASPI:
                    self.kit.stepper2.onestep(direction=2, style=1)
                self.theta1_real += MotionController.THETA_1_RADS_STEP
            else:
                if RASPI:
                    self.kit.stepper2.onestep(direction=1, style=1)
                self.theta1_real -= MotionController.THETA_1_RADS_STEP
        
        theta2_err, theta2_dir = self.shortest(self.theta2_target, self.theta2_real)
        if (np.abs(theta2_err) > MotionController.THETA_2_RADS_STEP):
            if theta2_dir:
                if RASPI:
                    self.kit.stepper1.onestep(direction=2, style=1)
                self.theta2_real += MotionController.THETA_2_RADS_STEP
            else:
                if RASPI:
                    self.kit.stepper1.onestep(direction=1, style=1)
                self.theta2_real -= MotionController.THETA_2_RADS_STEP
        
        if (np.abs(theta1_err) > MotionController.THETA_1_RADS_STEP) or (np.abs(theta2_err) > MotionController.THETA_2_RADS_STEP):
            return False
        else:
            return True
            

if __name__ == "__main__":
    controller = MotionController()

    theta_1_meas = 0
    theta_2_meas = 0

    controller.set_theta1(np.pi)
    controller.set_theta2(-np.pi)

    while not controller.tick(theta_1_meas, theta_2_meas):
        theta_1_meas = np.min([theta_1_meas + controller.THETA_1_RADS_STEP, np.pi])
        theta_2_meas = theta_2_meas - controller.THETA_2_RADS_STEP
        time.sleep(0.1)