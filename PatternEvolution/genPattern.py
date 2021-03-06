#!/usr/bin/python
#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import numpy
import numpy as np
import cv2
import time
import os

from pylab import imshow, show, ion

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from _functools import partial

import argparse

# for image rotation
import PIL
from PIL import Image
import math

# options parser to set all of these variables from the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--swatchSize', type=int, default=9)
parser.add_argument('--backgroundWidth', type=int, default=30)
parser.add_argument('--backgroundHeight', type=int, default=30)
parser.add_argument('--popSize', type=int, default=10)
parser.add_argument('--numGens', type=int, default=20000)
parser.add_argument('--backgroundType', type=int, default=0)
parser.add_argument('--random', type=int, default=64)
parser.add_argument('--numPics', type=int, default=1000)
parser.add_argument('--simple', action='store_true', default=False)
parser.add_argument('--isMax', action='store_true', default=False)
args = parser.parse_args()

# a switch... 0 for Translational, 1 for zoom, 2 for rotation
option = args.mode

# define the constraints of the swatch
SWATCH_NUM_PIXELS_WIDTH = args.swatchSize
SWATCH_NUM_PIXELS_HEIGHT = args.swatchSize

# define the constrains of the background
background_width = args.backgroundWidth
background_height = args.backgroundHeight

POPSIZE = args.popSize       # max number of individuals per generation
NUM_GENS = args.numGens      # max number of generations

seed = args.random
random.seed(seed)


# x dimension of OF: -1 for min, 1 for max
if(args.isMax):
    X_FUN = 1.0
else:
    X_FUN = -1.0

# for binary representations of Black and White (0,0,0) and (255,255,255)
BLACK = 0
WHITE = 255

DELTA = 1                                   # num pixels translated

NUM_PIX = args.numPics                      # 1 in 1000 individuals saved. 

ts = time.time()                            # create unique image directory
val = 0                                     # hack to keep track of generation number

os.makedirs('./Images/' + str(ts)[0:10] )   # save to new directory
f = open('./Images/' + str(ts)[0:10] + '/parameters.txt', 'w')

# OF of a white swatch moving on a black background 
# BASELINE = calculateBaselineTranslationalAvg();
BASELINE = 0

# 0 for all black, 2 for striped, 1 for random, 3 for traffic crossing
chooseBackground = args.backgroundType

# For each pixel, with probability indpb flip a coin to assign a value
# note that black -> black and white -> white 
def mutFlipPix_refactored(individual, indpb):
    for i in xrange(len(individual)):
        if random.random() < indpb:
            color = random.randint(0,1)
            if(color == 0):
                individual[i] = BLACK
            else:
                individual[i] = WHITE
    
    return individual,

# Generate a random image represented as a (pixels_height, pixels_width, 3) ndarray. 
# 50% black, 50% white (not independent pixel-coloring events)
def genRandomImage(pixels_height, pixels_width):
    im_rep = numpy.zeros((pixels_height,pixels_width,3), numpy.uint8)

    i = 0 
    while (i < 0.5 * pixels_height * pixels_width):
        x = random.randint(0, pixels_height - 1)
        y = random.randint(0, pixels_width - 1)
        if (im_rep[x][y][0] == BLACK):
            im_rep[x][y] = numpy.array([WHITE,WHITE,WHITE])
            i = i + 1

    return im_rep

# Generate a random image represented as a (pixels_height, pixels_width, 3) ndarray. 
# Alternate rows of black pixels with rows of white pixels. 
def genStripedImage(pixels_height, pixels_width):
    im_rep = numpy.zeros((pixels_height,pixels_width,3), numpy.uint8)

    for x in range (0, pixels_height):
        for y in range(0, pixels_width):
            if (x % 2 == 0):
                im_rep[x][y] = numpy.array([WHITE,WHITE,WHITE])
            else:
                im_rep[x][y] = numpy.array([BLACK,BLACK,BLACK])

    return im_rep


# background is a background_height x background_width sized-image. 

# CHOOSE 1 of the three: all black, random, or striped
if (chooseBackground == 0):
    background = numpy.zeros((background_height,background_width,3), numpy.uint8)
elif (chooseBackground == 1):
    background = genRandomImage(background_height, background_width)
elif (chooseBackground == 2):
    background = genStripedImage(background_height, background_width)
else:
    background = cv2.imread('traffic_bw.jpg')
    background_width = 320
    background_height = 240

# attribute generation
# to generate the first individual, called SWATCH_NUM_PIXELS_WIDTH * SWATCH_NUM_PIXELS_HEIGHT times
def gen_random_pixel_refactored(): 
    color = random.randint(0,1) # returns 0 or 1 for b or w 
    if (color == 1):
        color = WHITE
    return color 

# Fitness Function assignment. Calls base.Fitness, bound to the evalMax function below
creator.create("FitnessMax", base.Fitness, weights=(X_FUN,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox() 
history = tools.History()

# DEAP setup.
# attribute - random pixel (black or white)
# individual - list of x*y pixels, first selected by calling attribute x*y times
# population - type: list of individuals. 
toolbox.register("attr", gen_random_pixel_refactored)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=SWATCH_NUM_PIXELS_WIDTH * SWATCH_NUM_PIXELS_HEIGHT)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Given an individual, translate it across the background and return the average OF value as an (x,y) tuple
def calculatedTranslationalAvg_refactored(individual):
    # First definition of swatch -- empty. 3D Matrix. All pixels later defined. 
    swatch = numpy.empty((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
    
    # test if xover is breaking the representation by setting swatch to all grey 
    #for i in range(0,SWATCH_NUM_PIXELS_HEIGHT):
    #    for j in range(0,SWATCH_NUM_PIXELS_WIDTH):
    #        swatch[i][j] = numpy.array([100, 100, 100])
    
    # Convert list of pixels (an individual) to a swatch (3D matrix)
    for i in range(0, SWATCH_NUM_PIXELS_HEIGHT):
        for j in range (0, SWATCH_NUM_PIXELS_WIDTH):
            swatch[i][j] = numpy.array([individual[SWATCH_NUM_PIXELS_HEIGHT * i + j],individual[SWATCH_NUM_PIXELS_HEIGHT * i + j], individual[SWATCH_NUM_PIXELS_HEIGHT * i + j]])

    xFlowTotal = 0 
    yFlowTotal = 0    
    trials = 0 
    # SHOULD BE: 
    for k in range(0, background_width - SWATCH_NUM_PIXELS_WIDTH):
    
    # TEMPORARILY: one-pixel change for quick testing.  
    #for k in range(10, 11):  
        trials = trials + 1
        # create prev_im_rep, with the swatch at position k in the x dimension  
        prev_im_rep = background.copy()

        prev_im_rep[background_height/2 - SWATCH_NUM_PIXELS_HEIGHT/2:
                background_height/2 - SWATCH_NUM_PIXELS_HEIGHT/2 + SWATCH_NUM_PIXELS_HEIGHT, 
                k:SWATCH_NUM_PIXELS_WIDTH+k] = swatch

        # update position information
        k = k + DELTA

        # create im_rep_next, with the swatch at position k + DELTA in the x dimension  
        im_rep_next = background.copy()
        im_rep_next[background_height/2 - SWATCH_NUM_PIXELS_HEIGHT/2:
                background_height/2 - SWATCH_NUM_PIXELS_HEIGHT/2 + SWATCH_NUM_PIXELS_HEIGHT, 
                k:SWATCH_NUM_PIXELS_WIDTH+k] = swatch

        # make CV happy with grayscale images for previous and next frames
        prv = cv2.cvtColor(prev_im_rep, cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(im_rep_next, cv2.COLOR_BGR2GRAY) 

        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prv, nxt, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # sum our optical flow and then get averages for x and y direction
        total = cv2.sumElems(flow)
        xFlowTotal += total[0] / (background.shape[0]*background.shape[1])
        yFlowTotal += total[1] / (background.shape[0]*background.shape[1])

    # return tuple of average OF 
    return (xFlowTotal/trials, yFlowTotal/trials)

# this function returns the average value of the optical flow
def getAverageOpticalFlow(prev_frame, next_frame):
    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # sum our optical flow and then get averages for x and y direction
    total = cv2.sumElems(flow)
    xFlowAvg = total[0] / (background.shape[0]*background.shape[1])
    yFlowAvg = total[1] / (background.shape[0]*background.shape[1])

    flow_vector = [xFlowAvg, yFlowAvg]
    return flow_vector

# this function splits up images into 4 subimages to determine optical flow on
# after determining optical flow, of the 4 images, it computes a value of zoom
# based on the opposing vectors
def getZoomOpticalFlowSimple(prev_frame, next_frame):
    # store flow for each frame
    frame_flow = [[0,0],[0,0]]

    # scale the optic flow to show up...
    flowscale = 100

    # split into 9 frames
    for i in range(0,2):
        for j in range(0,2):
            left = j*prev_frame.shape[0]/2
            right = (j+1)*prev_frame.shape[0]/2
            top = i*prev_frame.shape[1]/2
            bottom = (i+1)*prev_frame.shape[1]/2

            sub_prev_frame = prev_frame[left:right, top:bottom]
            sub_next_frame = next_frame[left:right, top:bottom]

            #check optical flow for subframe
            frame_flow[i][j] = getAverageOpticalFlow(sub_prev_frame, sub_next_frame)
            
            midpoint = ((2*j+1)*prev_frame.shape[1]/4, (2*i+1)*prev_frame.shape[0]/4)
            flowpoint = (midpoint[0] + int(frame_flow[i][j][0]*flowscale), midpoint[1] + int(frame_flow[i][j][1]*flowscale))

    # create vectors to the center point from the center of the 9 frames
    tl_center = (1*prev_frame.shape[0]/4, 1*prev_frame.shape[1]/4)
    tr_center = (3*prev_frame.shape[0]/4, 1*prev_frame.shape[1]/4)
    br_center = (3*prev_frame.shape[0]/4, 3*prev_frame.shape[1]/4)
    bl_center = (1*prev_frame.shape[0]/4, 3*prev_frame.shape[1]/4)

    # look at all values and take the dot product with a vector to the center
    tl = np.dot(tl_center, frame_flow[0][0])
    tr = np.dot(tr_center, frame_flow[0][1])
    br = np.dot(br_center, frame_flow[1][1])
    bl = np.dot(bl_center, frame_flow[1][0])

    return tl + tr + br + bl # sum all of the divergence

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

    # create vectors to the center point from the center of the 9 frames
    center = (prev_frame.shape[0]/2, prev_frame.shape[1]/2)
    tl_center = (1*prev_frame.shape[0]/6, 1*prev_frame.shape[1]/6)
    t_center  = (3*prev_frame.shape[0]/6, 1*prev_frame.shape[1]/6)
    tr_center = (5*prev_frame.shape[0]/6, 1*prev_frame.shape[1]/6)
    r_center  = (5*prev_frame.shape[0]/6, 3*prev_frame.shape[1]/6)
    br_center = (5*prev_frame.shape[0]/6, 5*prev_frame.shape[1]/6)
    b_center  = (3*prev_frame.shape[0]/6, 5*prev_frame.shape[1]/6)
    bl_center = (1*prev_frame.shape[0]/6, 5*prev_frame.shape[1]/6)
    l_center  = (1*prev_frame.shape[0]/6, 3*prev_frame.shape[1]/6)

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


def calculatedZoomAvg(individual):
    center = [background.shape[0]/2, background.shape[1]/2]

    im_rep = background.copy()
    prev_im_rep = background.copy()

    swatch = numpy.empty((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
    
    for i in range(0, SWATCH_NUM_PIXELS_HEIGHT):
        for j in range (0, SWATCH_NUM_PIXELS_WIDTH):
            swatch[i][j] = numpy.array([individual[SWATCH_NUM_PIXELS_HEIGHT * i + j],individual[SWATCH_NUM_PIXELS_HEIGHT * i + j], individual[SWATCH_NUM_PIXELS_HEIGHT * i + j]])

    #imshow(prev_im_rep)
    #show()  

    # ZOOM
    #----------------------------------------------------------------------
    # zoom the image a pixel in size

    ## ARE THESE INDEXED CORRECTLY??? 
    # square so doesn't matter for now
    i_x = 10
    i_y = 10

    dim = (i_x, i_y)

    swatch1 = cv2.resize(swatch, dim, interpolation = cv2.INTER_AREA)


    left = center[0]-swatch1.shape[0]/2
    right = center[0]+swatch1.shape[0]/2
    top = center[1]-swatch1.shape[1]/2
    bottom = center[1]+swatch1.shape[1]/2

    prev_im_rep[left:right, top:bottom] = swatch1


    i_x = i_x + 10   #increment 2 pixels so the canvas grows a pixel around each frame
    i_y = i_y + 10 
    dim = (i_x, i_y)
     
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

    if(args.simple):
        zoom_val = getZoomOpticalFlowSimple(prv, nxt)
    else:
        zoom_val = getZoomOpticalFlow(prv, nxt)
    return zoom_val

# this function splits up images into 9 subimages to determine optical flow on
# after determining optical flow, of the 9 images, it computes a value of zoom
# based on the opposing vectors
def getRotationalOpticalFlow(prev_frame, next_frame):

    flow_rep = background.copy()
    
    # store flow for each frame
    frame_flow = [[0,0,0],[0,0,0],[0,0,0]]

    # scale the optic flow to show up...
    flowscale = 10

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
    tl_center = (1*prev_frame.shape[0]/6, 1*prev_frame.shape[1]/6)
    t_center  = (3*prev_frame.shape[0]/6, 1*prev_frame.shape[1]/6)
    tr_center = (5*prev_frame.shape[0]/6, 1*prev_frame.shape[1]/6)
    r_center  = (5*prev_frame.shape[0]/6, 3*prev_frame.shape[1]/6)
    br_center = (5*prev_frame.shape[0]/6, 5*prev_frame.shape[1]/6)
    b_center  = (3*prev_frame.shape[0]/6, 5*prev_frame.shape[1]/6)
    bl_center = (1*prev_frame.shape[0]/6, 5*prev_frame.shape[1]/6)
    l_center  = (1*prev_frame.shape[0]/6, 3*prev_frame.shape[1]/6)

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

# this function splits up images into 4 subimages to determine optical flow on
# after determining optical flow, of the 4 images, it computes a value of zoom
# based on the opposing vectors
def getRotationalOpticalFlowSimple(prev_frame, next_frame):

    flow_rep = background.copy()
    
    # store flow for each frame
    frame_flow = [[0,0],[0,0]]

    # scale the optic flow to show up...
    flowscale = 10

    # split into 4 frames
    for i in range(0,2):
        for j in range(0,2):
            left = j*prev_frame.shape[0]/2
            right = (j+1)*prev_frame.shape[0]/2
            top = i*prev_frame.shape[1]/2
            bottom = (i+1)*prev_frame.shape[1]/2

            sub_prev_frame = prev_frame[left:right, top:bottom]
            sub_next_frame = next_frame[left:right, top:bottom]

            #check optical flow for subframe
            frame_flow[i][j] = getAverageOpticalFlow(sub_prev_frame, sub_next_frame)
            
            midpoint = ((2*j+1)*prev_frame.shape[1]/4, (2*i+1)*prev_frame.shape[0]/4, )
            flowpoint = (midpoint[0] + int(frame_flow[i][j][0]*flowscale), midpoint[1] + int(frame_flow[i][j][1]*flowscale))

            #draw the flow to our visualizer
            cv2.line(flow_rep, midpoint, flowpoint, (0,255,0),1)

    # create vectors to the center point from the center of the 9 frames
    tl_center = (1*prev_frame.shape[0]/4, 1*prev_frame.shape[1]/4)
    tr_center = (3*prev_frame.shape[0]/4, 1*prev_frame.shape[1]/4)
    br_center = (3*prev_frame.shape[0]/4, 3*prev_frame.shape[1]/4)
    bl_center = (1*prev_frame.shape[0]/4, 3*prev_frame.shape[1]/4)

    # look at all values and take the cross product with a vector to the center
    tl = np.cross(tl_center, frame_flow[0][0])
    tr = np.cross(tr_center, frame_flow[0][1])
    br = np.cross(br_center, frame_flow[1][1])
    bl = np.cross(bl_center, frame_flow[1][0])

    return tl + tr + br + bl


# TO-DO 
def calculatedRotAvg(individual):
    im_rep = background.copy()
    prev_im_rep = background.copy()

    swatch = numpy.empty((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
    
    for i in range(0, SWATCH_NUM_PIXELS_HEIGHT):
        for j in range (0, SWATCH_NUM_PIXELS_WIDTH):
            swatch[i][j] = numpy.array([individual[SWATCH_NUM_PIXELS_HEIGHT * i + j],individual[SWATCH_NUM_PIXELS_HEIGHT * i + j], individual[SWATCH_NUM_PIXELS_HEIGHT * i + j]])

    i = 0; 

    # max shift value for offset during rotation
    shift = (math.sqrt(2) - 1.0)/2.0

    # rotate the swatch
    angle = i
    left = int(im_rep.shape[1]/2 - (swatch.shape[1]/2) - (shift * swatch.shape[1]) * math.cos(math.pi*(((angle*2)%180-90)/180.0)))
    top = int(im_rep.shape[0]/2 - (swatch.shape[0]/2) - (shift * swatch.shape[0]) * math.sin(math.pi*(((angle*2)%180)/180.0)))
    src_im = im = Image.fromarray(swatch)
    dst_im = Image.fromarray(background)
    im = src_im.convert('RGBA')
    rot = im.rotate( angle, expand=2, resample=PIL.Image.BICUBIC) # NEAREST, BILINEAR, BICUBIC
    dst_im.paste( rot, (left, top), rot )
    dst_im_rgb = dst_im.convert('RGB')
    prev_im_rep = np.asarray(dst_im_rgb)

    
    # move the image i pixels to the right
    # TEMP: do this only once! 
    i = i + 10
    #if i >= 360:
    #    i = 0

    # rotate the swatch
    angle = i
    left = int(im_rep.shape[1]/2 - (swatch.shape[1]/2) - (shift * swatch.shape[1]) * math.cos(math.pi*(((angle*2)%180-90)/180.0)))
    top = int(im_rep.shape[0]/2 - (swatch.shape[0]/2) - (shift * swatch.shape[0]) * math.sin(math.pi*(((angle*2)%180)/180.0)))
    src_im = im = Image.fromarray(swatch)
    dst_im = Image.fromarray(background)
    im = src_im.convert('RGBA')
    rot = im.rotate( angle, expand=2, resample=PIL.Image.BICUBIC) # NEAREST, BILINEAR, BICUBIC
    dst_im.paste( rot, (left, top), rot )
    dst_im_rgb = dst_im.convert('RGB')
    im_rep = np.asarray(dst_im_rgb)
    
    # make CV happy with grayscale images for previous and next frames
    prv = cv2.cvtColor(prev_im_rep, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(im_rep, cv2.COLOR_BGR2GRAY) 

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prv, nxt, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    if(args.simple):
        rot_val = getRotationalOpticalFlowSimple(prv, nxt)  # simplified to 4 grid
    else:
        rot_val = getRotationalOpticalFlow(prv, nxt)  # simplified to 4 grid

    return rot_val

# Returns the computed OF value for fitness calculations
# depending on which 'option' is set 
def evalMax(individual):
    swatch = numpy.empty((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
    
    for i in range(0, SWATCH_NUM_PIXELS_HEIGHT):
        for j in range (0, SWATCH_NUM_PIXELS_WIDTH):
            swatch[i][j] = numpy.array([individual[SWATCH_NUM_PIXELS_HEIGHT * i + j],individual[SWATCH_NUM_PIXELS_HEIGHT * i + j], individual[SWATCH_NUM_PIXELS_HEIGHT * i + j]])

    # hack to see which generation we're on. val is set to 0 as start up, 
    global val 
    global f
    # when val is first incremented, we create a directory. 
    if (val == 0):
        f.write("Mode (0 trans, 1 zoom, 2 rotate): " + str(option) + '\n')
        f.write("Swatch Size:" + str(SWATCH_NUM_PIXELS_HEIGHT) + 'by' + str(SWATCH_NUM_PIXELS_WIDTH) + '\n')
        f.write("Background Size:" + str(background_height) + 'by' + str(background_width) + '\n')
        f.write("Background Type (0 black, 1 striped, 2 random):" + str(chooseBackground) + '\n')
        f.write("Popsize:" + str(POPSIZE) + '\n')
        f.write("Number of generations:" + str(NUM_GENS) + '\n')
        f.write("Min or max:" + str(X_FUN) + '\n')
        f.write("Random seed:" + str(seed) + '\n')
        f.write("y_cood of motion:" + str(background_height/2 - SWATCH_NUM_PIXELS_HEIGHT/2) + '\n')
        f.write("x_cood of motion (range):" + str(0) + "-" + str(background_width - SWATCH_NUM_PIXELS_WIDTH) + '\n')


        cv2.imwrite('./Images/' + str(ts)[0:10]  + '/pic_trans_BACKGROUND.png', background)

    if (option == 0):
        (h,w) = calculatedTranslationalAvg_refactored(individual)
    elif (option == 1):
        h = calculatedZoomAvg(individual)
    else:
        h = calculatedRotAvg(individual)

    # for every NUM_PIX image generated from then on, we save that image 
    if (val % NUM_PIX == 0):
        cv2.imwrite('./Images/' + str(ts)[0:10]  + '/pic_trans_' + str(val) + '.png', swatch)
        f.write(str(val) + ', ' + str(h) + '\n')

    val = val + 1

    return h,

# cross over function -- provided by DEAP
def cx(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
    
toolbox.register("evaluate", evalMax)
toolbox.register("mate", cx)
toolbox.register("mutate", mutFlipPix_refactored, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Decorate the variation operators for history usage only
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

def main():
    pop = toolbox.population(n=POPSIZE)

    # Create the population and populate the history
    history.update(pop)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=numpy.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    try: 
        # take 3 best individuals per generation, create up to POPSIZE more 
        # mutate with prob mutpb
        # cross-over with prob cxpb
        algorithms.eaMuPlusLambda(pop, toolbox, mu=3, lambda_=POPSIZE, cxpb=0.25, mutpb=0.5, ngen=NUM_GENS, stats=stats, halloffame=hof, verbose=True)
        #algorithms.eaSimple(pop, toolbox, cxpb=0.25, mutpb=0.5, ngen=NUM_GENS, stats=stats, halloffame=hof, verbose=True)
    finally: 
        # print the final picture
        prev_im_rep = background.copy()

        # get the swatch from the hall of fame
        PIC_new = numpy.zeros((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
        for i in range(0, SWATCH_NUM_PIXELS_HEIGHT):
            for j in range (0, SWATCH_NUM_PIXELS_WIDTH):
                PIC_new[i][j] = numpy.array([hof[0][SWATCH_NUM_PIXELS_HEIGHT * i + j],hof[0][SWATCH_NUM_PIXELS_HEIGHT * i + j], hof[0][SWATCH_NUM_PIXELS_HEIGHT * i + j]])

        # overlay the swatch on the background
        prev_im_rep[background_height/2 - SWATCH_NUM_PIXELS_HEIGHT/2:
                background_height/2 - SWATCH_NUM_PIXELS_HEIGHT/2 + SWATCH_NUM_PIXELS_HEIGHT, 
                10:SWATCH_NUM_PIXELS_WIDTH+10] = PIC_new

        #save
        x = evalMax(hof[0])
        f.write('FINAL, ' + str(x[0]) + '\n')
        cv2.imwrite('./Images/' + str(ts)[0:10]  + '/pic_FINAL.png', PIC_new)
        
    return pop, stats, hof

if __name__ == "__main__":
    main(); 
