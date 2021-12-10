from adafruit_motorkit import MotorKit
import time

kit = MotorKit()

while True:
    for i in range(100):
        kit.stepper1.onestep(direction=1, style=2)
        kit.stepper2.onestep(direction=2, style=2) # DUAL COIL is Higher Torque
        time.sleep(0.1)
    for i in range(100):
        kit.stepper1.onestep(direction=2, style=2)
        kit.stepper2.onestep(direction=1, style=2) # DUAL COIL is Higher Torque
        time.sleep(0.1)