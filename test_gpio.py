# approach 1

from time import sleep
import RPi.GPIO as GPIO
controlPin = 26
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(True)
GPIO.setup(controlPin, GPIO.OUT, initial=0)
while True:
  print("on")
  GPIO.output(controlPin, True)
  # led.on()
  sleep(1)
  print("off")
  # led.off()
  GPIO.output(controlPin, False)
  sleep(1)

# approach 2

from time import sleep
from gpiozero import LED
controlPin = 26
led = LED(controlPin)
while True:
  print("on")
  led.on()
  sleep(1)
  print("off")
  led.off()
  sleep(1)