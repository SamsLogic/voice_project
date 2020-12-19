import RPi.GPIO as GPIO
import time


def head_servo_movement(direction_x,direction_y):
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(27,GPIO.OUT)
    GPIO.setup(17,GPIO.OUT)

    servox = GPIO.PWM(17,50)
    servoy = GPIO.PWM(27,50)

    servoy.start(0)
    servox.start(0)
    
    servox_angle = (direction_x)*(direction_x > 90 and direction_x<=180) + (direction_x-180) * (direction_x > 180 and direction_x<=270) + 180 * (direction_x>=0 and direction_x<=90) + (0)*(direction_x>270)
    servoy_angle = 180*(direction_y>90 and direction_y<=180) + 0 * (direction_y <=270 and direction_y>180) + (direction_y-270) * (direction_y > 270 and direction_y < 360) + (90-direction_y) * (direction_y >=0 and direction_y<90)
    print(servox_angle,servoy_angle)
    servox.ChangeDutyCycle(2+((servox_angle)/18))
    servoy.ChangeDutyCycle(2+(servoy_angle)/18)
    time.sleep(0.4)
    servox.ChangeDutyCycle(0)
    servoy.ChangeDutyCycle(0)
    servox.stop()
    servoy.stop()
