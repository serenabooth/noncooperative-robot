# Black square move CV test
import numpy as np
import cv2
import random
from pylab import imshow, show

swatch_width = 50
swatch_height = 50

background_width = 320
background_height = 240

midpoint = (int(background_width/2), int(background_height/2))

# create image representation
im_rep = np.zeros((background_height,background_width,3), np.uint8)
prev_im_rep = im_rep
# for displaying the flow
flow_rep = np.zeros((background_height,background_width,3), np.uint8)
full_rep = np.zeros((background_height,background_width*2,3), np.uint8)

# create image background
background = np.zeros((background_height,background_width,3), np.uint8)

# create white swatch
swatch = np.empty((swatch_width,swatch_width,3), np.uint8)
for i in range(0,swatch_height):
	for j in range(0,swatch_width):
		swatch[i][j] = (255,255,255)

# # create spotted swatch
# for i in range(0,swatch_height):
# 	for j in range(0,swatch_width):
# 		if(i%3 == 0 and j%3 == 0):
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
	flow_rep = background.copy()

	# replace a portion of the array with the swatch
	y_pos = background_height/2 - swatch_height/2
	im_rep[y_pos:y_pos+swatch.shape[1], i:swatch.shape[0]+i] = swatch
	
	# move the image i pixels to the right
	i = i + 2
	if i > (background.shape[1]-swatch.shape[1]):
		i = 0

	# make CV happy with grayscale images for previous and next frames
	prv = cv2.cvtColor(prev_im_rep, cv2.COLOR_BGR2GRAY)
	nxt = cv2.cvtColor(im_rep, cv2.COLOR_BGR2GRAY) 

	# calculate optical flow
 	flow = cv2.calcOpticalFlowFarneback(prv, nxt, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
	
	flowscale = 1
	spacing = 4

 	# draw flow lines every n pixels
 	for row in range(0, prv.shape[0], spacing):
 		for col in range(0, prv.shape[1], spacing):
 			x = flow[row][col][0]
 			y = flow[row][col][1]

 			midpoint = (row, col)
 			# print "midpoint: "
 			# print midpoint
			flowpoint = (midpoint[0] + int(x*flowscale), midpoint[1] + int(y*flowscale))
			# print "flowpoint: "
			# print flowpoint

			midpoint = (midpoint[1], midpoint[0])
			flowpoint = (flowpoint[1], flowpoint[0])
			# canvas, from, to, color (bgra), weight
			cv2.line(flow_rep, midpoint, flowpoint, (255,255,0),1)
	
	full_rep[0:im_rep.shape[0], 0:im_rep.shape[1]] = im_rep
	full_rep[0:im_rep.shape[0], im_rep.shape[1]:2*im_rep.shape[1]] = flow_rep
	# draw our sweet animation
	cv2.imshow('animation', full_rep)

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
	xFlowAvg = total[0] / (im_rep.shape[0]*im_rep.shape[1])
	yFlowAvg = total[1] / (im_rep.shape[0]*im_rep.shape[1])

	flow_vector = [xFlowAvg, yFlowAvg]
	return flow_vector

#
def getTotalOpticalFlowOverAnimation():
	print "not implemented yet"

