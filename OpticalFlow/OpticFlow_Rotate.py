# Black square move CV test
import numpy as np
import cv2
import random
from pylab import imshow, show
import PIL
from PIL import Image
import math

swatch_width = 150
swatch_height = 150

background_width = 320
background_height = 240

shift = (math.sqrt(2) - 1.0)/2.0

midpoint = (int(background_width/2), int(background_height/2))

# create image representation
im_rep = np.zeros((background_height,background_width,3), np.uint8)
prev_im_rep = im_rep
# for displaying the flow
flow_rep = np.zeros((background_height,background_width,3), np.uint8)
full_rep = np.zeros((background_height,background_width*2,3), np.uint8)

# create image background
background = np.zeros((background_height,background_width,3), np.uint8)
black = np.zeros((background_height,background_width,3), np.uint8)

# create white swatch
swatch = np.empty((swatch_width,swatch_width,3), np.uint8)
for i in range(0,swatch_height):
	for j in range(0,swatch_width):
		if(i == swatch_width/2 and j == swatch_height/2):	#show the center 
			swatch[i][j] = (0,0,0)
		else:
			swatch[i][j] = (255,255,255)


# # create spotted swatch
# for i in range(0,swatch_height):
# 	for j in range(0,swatch_width):
# 		if(i%3 == 0 and j%3 == 0):
# 			swatch[i][j] = (255,255,255)
# 		else:
# 			swatch[i][j] = (0,0,0)

# create spotted background
for i in range(0,background_height):
	for j in range(0,background_width):
		if(i%3 == 0 and j%3 == 0):
			background[i][j] = (255,255,255)
		else:
			background[i][j] = (0,0,0)

		# prev_im_rep[i][j] = (0,0,0)

# this function returns the average value of the optical flow
def getAverageOpticalFlow(prev_frame, next_frame):
	# calculate optical flow
 	flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

 	# sum our optical flow and then get averages for x and y direction
	total = cv2.sumElems(flow)
	xFlowAvg = total[0] / (im_rep.shape[0]*im_rep.shape[1])
	yFlowAvg = total[1] / (im_rep.shape[0]*im_rep.shape[1])

	flow_vector = [xFlowAvg, yFlowAvg]
	return flow_vector

# this function splits up images into 9 subimages to determine optical flow on
# after determining optical flow, of the 9 images, it computes a value of zoom
# based on the opposing vectors
def getRotationalOpticalFlow(prev_frame, next_frame):
	
	# store flow for each frame
	frame_flow = [[0,0,0],[0,0,0],[0,0,0]]

	# scale the optic flow to show up...
	flowscale = 100

	# print "Flow"
	# split into 9 frames
	for i in range(0,3):
		for j in range(0,3):
			left = j*prev_frame.shape[0]/3
			right = (j+1)*prev_frame.shape[0]/3
			top = i*prev_frame.shape[1]/3
			bottom = (i+1)*prev_frame.shape[1]/3

			#print "bounds: (%d, %d, %d, %d)" % (left, right, top, bottom)
			sub_prev_frame = prev_frame[left:right, top:bottom]
			sub_next_frame = next_frame[left:right, top:bottom]
			# print "subframe: (%d, %d)" % (sub_prev_frame.shape[0], sub_prev_frame.shape[1])

			#check optical flow for subframe
			frame_flow[i][j] = getAverageOpticalFlow(sub_prev_frame, sub_next_frame)
			#print "FrameFlow (%d, %d)" % (100*frame_flow[i][j][0], 100*frame_flow[i][j][1])

 			midpoint = ((2*j+1)*prev_frame.shape[1]/6, (2*i+1)*prev_frame.shape[0]/6, )
			flowpoint = (midpoint[0] + int(frame_flow[i][j][0]*flowscale), midpoint[1] + int(frame_flow[i][j][1]*flowscale))
			# print "MidPoint (%d, %d)" % (midpoint[0], midpoint[1])
			# print "FlowPoint (%d, %d)" % (flowpoint[0], flowpoint[1])
			# midpoint = (midpoint[1], midpoint[0])
			# flowpoint = (flowpoint[1], flowpoint[0])

			#draw the flow to our visualizer
			cv2.line(flow_rep, midpoint, flowpoint, (0,255,0),1)

	# create vectors to the center point from the center of the 9 frames
	tl_center = (1*prev_frame.shape[0]/6, 1*prev_frame.shape[1]/6)
	t_center  = (3*prev_frame.shape[0]/6, 1*prev_frame.shape[1]/6)
	tr_center = (5*prev_frame.shape[0]/6, 1*prev_frame.shape[1]/6)
	r_center  = (5*prev_frame.shape[0]/6, 3*prev_frame.shape[1]/6)
	br_center = (5*prev_frame.shape[0]/6, 5*prev_frame.shape[1]/6)
	b_center  = (3*prev_frame.shape[0]/6, 5*prev_frame.shape[1]/6)
	bl_center = (1*prev_frame.shape[0]/6, 5*prev_frame.shape[1]/6)
	l_center  = (1*prev_frame.shape[0]/6, 3*prev_frame.shape[1]/6)

	#print all frame flow
	# print "9 section flow"
	# print frame_flow[0][0]
	# print frame_flow[0][1]
	# print frame_flow[0][2]
	# print frame_flow[1][0]
	# print frame_flow[1][1]
	# print frame_flow[1][2]
	# print frame_flow[2][0]
	# print frame_flow[2][1]
	# print frame_flow[2][2]

	# look at all values and take the cross product with a vector to the center
	tl = np.cross(tl_center, frame_flow[0][0])
	t  = np.cross(t_center,  frame_flow[0][1])
	tr = np.cross(tr_center, frame_flow[0][2])
	r  = np.cross(r_center,  frame_flow[1][2])
	br = np.cross(br_center, frame_flow[2][2])
	b  = np.cross(b_center,  frame_flow[2][1])
	bl = np.cross(bl_center, frame_flow[2][0])
	l  = np.cross(l_center,  frame_flow[1][0])

	return tl + t + tr + r + br + b + bl + l


#index to increment our movement
i = 0 

while True:

	# initialize the image to black
	im_rep = background.copy()
	flow_rep = black.copy()

	# rotate the swatch
	# rotate the image by n degrees
	angle = i
	left = int(im_rep.shape[1]/2 - (swatch.shape[1]/2) - (shift * swatch.shape[1]) * math.cos(math.pi*(((angle*2)%180-90)/180.0)))
	top = int(im_rep.shape[0]/2 - (swatch.shape[0]/2) - (shift * swatch.shape[0]) * math.sin(math.pi*(((angle*2)%180)/180.0)))
	src_im = im = Image.fromarray(swatch)
	dst_im = Image.fromarray(background)
	im = src_im.convert('RGBA')
	rot = im.rotate( angle, expand=2, resample=PIL.Image.BICUBIC) # NEAREST, BILINEAR, BICUBIC
	dst_im.paste( rot, (left, top), rot )
	dst_im_rgb = dst_im.convert('RGB')
	# dst_im_rgb.show()
	im_rep = np.asarray(dst_im_rgb)

	# move the image i pixels to the right
	i = i + 5
	if i >= 360:
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
			cv2.line(flow_rep, midpoint, flowpoint, (0,0,255),1)

	zoom_val = getRotationalOpticalFlow(prv, nxt)
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