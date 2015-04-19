#!/usr/bin/env python

import numpy as np
import cv2
import serial

source = 0
cam = cv2.VideoCapture(source)

size = (320,240)
cam.set(3, size[0]) # video width setting
cam.set(4, size[1]) # video height setting

ret, prev = cam.read()

scale = 1

small = cv2.resize(prev, (0,0), fx=scale, fy=scale) 
prevgray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(small)
hsv[...,1] = 255

midpoint = (int(small.shape[1]/2), int(small.shape[0]/2))

print "Resizing from {},{} to {},{}".format(prev.shape[0], prev.shape[1], small.shape[0], small.shape[1])

# connect to Arduino on port...
# send command to Arduino to move forward 1000 steps at a speed of 150
ser = serial.Serial('/dev/tty.usbmodem1411', 9600)
ser.write('1 1000 150\n')


while True:
    ret, img = cam.read()

    # convert to grayscale and resize down by half
    small = cv2.resize(img, (0,0), fx=scale, fy=scale) 
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # calculate optical flow between previous and current frame
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    total = cv2.sumElems(flow)
    xFlowAve = total[0] / (small.shape[0]*small.shape[1])
    yFlowAve = total[1] / (small.shape[0]*small.shape[1])

    print "ave flow: {},{}".format(xFlowAve,yFlowAve)
    flowscale = 20
    flowpoint = (midpoint[0] + int(xFlowAve*flowscale), midpoint[1] + int(yFlowAve*flowscale))

    # from, to, color (bgra), weight
    cv2.line(small, midpoint, flowpoint, (0,255,0),2)
    cv2.imshow('optical_flow', small)

    
    # escape key to quit
    ch = 0xFF & cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()