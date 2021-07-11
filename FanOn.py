from __future__ import print_function
import time
from dual_max14870_rpi import motors, MAX_SPEED
from gpiozero import CPUTemperature

# Define a custom exception to raise if a fault is detected.
class DriverFault(Exception):
    pass

def raiseIfFault():
    if motors.getFault():
        raise DriverFault

try:
    motors.setSpeeds(0, 0)
    motors.enable()

    while(True):
        cpu = CPUTemperature()
        temp = cpu.temperature
        if(temp > 50):
            motors.motor1.setSpeed(MAX_SPEED)
        raiseIfFault()
        time.sleep(10)

except DriverFault:
    print("Driver fault!")

finally:
    # Stop the motors, even if there is an exception
    # or the user presses Ctrl+C to kill the process.
    motors.forceStop()
