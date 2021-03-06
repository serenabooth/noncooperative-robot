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

# this function splits up images into 9 subimages to determine optical flow on
# after determining optical flow, of the 9 images, it computes a value of zoom
# based on the opposing vectors
def getZoomOpticalFlow(prev_frame, next_frame):
	
	# store flow for each frame
	frame_flow = [[0,0,0],[0,0,0],[0,0,0]]

	# scale the optic flow to show up...
	flowscale = 100

	# split into 9 frames
	for i in range(0,3):
		for j in range(0,3):
			left = j*prev_frame.shape[0]/3
			right = (j+1)*prev_frame.shape[0]/3
			top = i*prev_frame.shape[1]/3
			bottom = (i+1)*prev_frame.shape[1]/3

			sub_prev_frame = prev_frame[left:right, top:bottom]
			sub_next_frame = next_frame[left:right, top:bottom]

			#check optical flow for subframe
			frame_flow[i][j] = getAverageOpticalFlow(sub_prev_frame, sub_next_frame)
			
 			midpoint = ((2*j+1)*prev_frame.shape[1]/6, (2*i+1)*prev_frame.shape[0]/6, )
			flowpoint = (midpoint[0] + int(frame_flow[i][j][0]*flowscale), midpoint[1] + int(frame_flow[i][j][1]*flowscale))

			# midpoint = (midpoint[1], midpoint[0])
			# flowpoint = (flowpoint[1], flowpoint[0])

			#draw the flow to our visualizer
			cv2.line(flow_rep, midpoint, flowpoint, (0,255,0),1)

	# create vectors to the center point from the center of the 9 frames
	center = (prev_frame.shape[0]/2, prev_frame.shape[1]/2)
	tl_center = (1*prev_frame.shape[0]/6 - center[0], 1*prev_frame.shape[1]/6 - center[1])
	t_center  = (3*prev_frame.shape[0]/6 - center[0], 1*prev_frame.shape[1]/6 - center[1])
	tr_center = (5*prev_frame.shape[0]/6 - center[0], 1*prev_frame.shape[1]/6 - center[1])
	r_center  = (5*prev_frame.shape[0]/6 - center[0], 3*prev_frame.shape[1]/6 - center[1])
	br_center = (5*prev_frame.shape[0]/6 - center[0], 5*prev_frame.shape[1]/6 - center[1])
	b_center  = (3*prev_frame.shape[0]/6 - center[0], 5*prev_frame.shape[1]/6 - center[1])
	bl_center = (1*prev_frame.shape[0]/6 - center[0], 5*prev_frame.shape[1]/6 - center[1])
	l_center  = (1*prev_frame.shape[0]/6 - center[0], 3*prev_frame.shape[1]/6 - center[1])

	# look at all values and take the dot product with a vector to the center
	tl = np.dot(tl_center, frame_flow[0][0])
	t  = np.dot(t_center,  frame_flow[0][1])
	tr = np.dot(tr_center, frame_flow[0][2])
	r  = np.dot(r_center,  frame_flow[1][2])
	br = np.dot(br_center, frame_flow[2][2])
	b  = np.dot(b_center,  frame_flow[2][1])
	bl = np.dot(bl_center, frame_flow[2][0])
	l  = np.dot(l_center,  frame_flow[1][0])

	return tl + r + tr + r + br + b + bl + l # summ all of the divergence


#index to increment our movement
i = 100 
center = [background.shape[0]/2, background.shape[1]/2]


while True:

	# initialize the image to black
	im_rep = background.copy()
	flow_rep = background.copy()

	# TRANSLATION
	#----------------------------------------------------------------------
	# # replace a portion of the array with the swatch
	# y_pos = background_height/2 - swatch_height/2
	# im_rep[y_pos:y_pos+swatch.shape[1], i:swatch.shape[0]+i] = swatch
	
	# # move the image i pixels to the right
	# i = i + 10
	# if i > (background.shape[1]-swatch.shape[1]):
	# 	i = 0
	#----------------------------------------------------------------------


	# ZOOM
	#----------------------------------------------------------------------
	# zoom the image a pixel in size
	i = i + 2	#increment 2 pixels so the canvas grows a pixel around each frame
	dim = (i, i)
	 
	# perform the actual resizing of the image and show it
	swatch = cv2.resize(swatch, dim, interpolation = cv2.INTER_AREA)

	# replace a portion of the array with the swatch
	left = center[0]-swatch.shape[0]/2
	right = center[0]+swatch.shape[0]/2
	top = center[1]-swatch.shape[1]/2
	bottom = center[1]+swatch.shape[1]/2

	im_rep[left:right, top:bottom] = swatch
	#----------------------------------------------------------------------

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
			cv2.line(flow_rep, midpoint, flowpoint, (0,0,255),1)

	zoom_val = getZoomOpticalFlow(prv, nxt)
	# print "Zoom: "
	# print zoom_val
	
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