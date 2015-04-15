# Black square move CV test
import numpy as np
import cv2
import random
from pylab import imshow, show

swatch_width = 100
swatch_height = 100

background_width = 320
background_height = 240

# create image representation
im_rep = np.zeros((background_height,background_width,3), np.uint8)
prev_im_rep = im_rep

# create image background
background = np.zeros((background_height,background_width,3), np.uint8)

# create white swatch
swatch = np.empty((swatch_width,swatch_width,3), np.uint8)
for i in range(0,swatch_height):
	for j in range(0,swatch_width):
		swatch[i][j] = (255,255,255)

# create spotted swatch
# for i in range(0,swatch_height):
# 	for j in range(0,swatch_width):
# 		if(i%5 == 0 and j%5 == 0):
# 			swatch[i][j] = (255,255,255)
# 		else:
# 			swatch[i][j] = (0,0,0)

# create black background
for i in range(0,background_height):
	for j in range(0,background_width):
		prev_im_rep[i][j] = (0,0,0)


#index to increment our movement
i = 0 

while True:

	# initialize the image to black
	im_rep = background.copy()

	# replace a portion of the array with the swatch
	y_pos = 70
	im_rep[y_pos:y_pos+swatch.shape[1], i:swatch.shape[0]+i] = swatch
	
	# move the image i pixels to the right
	i = i + 1

	# make CV happy with grayscale images for previous and next frames
	prv = cv2.cvtColor(prev_im_rep, cv2.COLOR_BGR2GRAY)
	nxt = cv2.cvtColor(im_rep, cv2.COLOR_BGR2GRAY) 

	# calculate optical flow
 	flow = cv2.calcOpticalFlowFarneback(prv, nxt, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

 	# sum our optical flow and then get averages for x and y direction
	total = cv2.sumElems(flow)
	xFlowAve = total[1] / (im_rep.shape[0]*im_rep.shape[1])
	yFlowAve = total[0] / (im_rep.shape[0]*im_rep.shape[1])

	print "ave flow: {},{}".format(xFlowAve,yFlowAve)

	# draw our sweet animation
	cv2.imshow('animation', im_rep)

	# set previous frame equal to current frame
	prev_im_rep = im_rep

	# escape key to quit (insert image of fire escape here)
	ch = 0xFF & cv2.waitKey(5)
	if ch == 27:
		break

cv2.destroyAllWindows()

# this function returns the average value of the optical flow
def getAverageOpticalFlow(prev_frame, next_frame):
	# calculate optical flow
 	flow = cv2.calcOpticalFlowFarneback(prv, nxt, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

 	# sum our optical flow and then get averages for x and y direction
	total = cv2.sumElems(flow)
	xFlowAvg = total[1] / (im_rep.shape[0]*im_rep.shape[1])
	yFlowAvg = total[0] / (im_rep.shape[0]*im_rep.shape[1])

	flow_vector = [xFlowAvg, yFlowAvg]
	return flow_vector

#
def getTotalOpticalFlowOverAnimation():
	print "not implemented yet"

