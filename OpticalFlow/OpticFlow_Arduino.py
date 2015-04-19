#!/usr/bin/env python

import numpy as np
import cv2
import video
import serial

if __name__ == '__main__':
  import sys
  try:
      fn = sys.argv[1]
  except:
      fn = 0

  cam = video.create_capture(fn)
  # capture one frame and convert to gray
  # before we start the loop so we always have
  # 2 frames needed for oflow calculation
  # also prevgray var will persist in loop
  ret, prev = cam.read()

  scale = 0.25
  
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
      xFlowAve = total[1] / (small.shape[0]*small.shape[1])
      yFlowAve = total[0] / (small.shape[0]*small.shape[1])

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