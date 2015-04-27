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

# 0 for Translational, 1 for zoom, 2 for rotation
option = 1 

SWATCH_NUM_PIXELS_WIDTH = 12
SWATCH_NUM_PIXELS_HEIGHT = 12

background_width = 320
background_height = 240

POPSIZE = 10
NUM_GENS = 5000
#x dimension of OF: -1 for min, 1 for max
X_FUN = 1.0 

BLACK = 0
WHITE = 255

DELTA = 1 #num pixels translated

ts = time.time() 
val = 0
# OF of a white swatch moving on a black background 
#BASELINE = calculateBaselineTranslationalAvg();
BASELINE = 0

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
# 50% black, 50% white
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
# 50% black, 50% white
def genStripedImage(pixels_height, pixels_width):
    im_rep = numpy.zeros((pixels_height,pixels_width,3), numpy.uint8)

    for x in range (0, pixels_height):
        for y in range(0, pixels_width):
            if (x % 3 == 0 or x % 4 == 0):
                im_rep[x][y] = numpy.array([WHITE,WHITE,WHITE])
            else:
                im_rep[x][y] = numpy.array([BLACK,BLACK,BLACK])

    return im_rep


#background is a 240x320 all-black image. 
background = numpy.zeros((background_height,background_width,3), numpy.uint8)
#background = genRandomImage(background_height, background_width)
#background = genStripedImage(background_height, background_width)

#imshow(background)
#show()
def gen_random_pixel_refactored(): 
    color = random.randint(0,1) # returns 0 or 1 for b or w 
    if (color == 1):
        color = WHITE
    return color 

creator.create("FitnessMax", base.Fitness, weights=(X_FUN,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox() 
history = tools.History()

toolbox.register("attr", gen_random_pixel_refactored)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=SWATCH_NUM_PIXELS_WIDTH * SWATCH_NUM_PIXELS_HEIGHT)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def calculatedTranslationalAvg_refactored(individual):

    swatch = numpy.empty((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
    
    # test if xover is breaking the representation
    #for i in range(0,SWATCH_NUM_PIXELS_HEIGHT):
    #    for j in range(0,SWATCH_NUM_PIXELS_WIDTH):
    #        swatch[i][j] = numpy.array([100, 100, 100])
    
    for i in range(0, SWATCH_NUM_PIXELS_HEIGHT):
        for j in range (0, SWATCH_NUM_PIXELS_WIDTH):
            swatch[i][j] = numpy.array([individual[SWATCH_NUM_PIXELS_HEIGHT * i + j],individual[SWATCH_NUM_PIXELS_HEIGHT * i + j], individual[SWATCH_NUM_PIXELS_HEIGHT * i + j]])

    xFlowTotal = 0 
    yFlowTotal = 0    

    global val 
    val = val + 1

    if (val == 1):
        os.makedirs('./Images/' + str(ts)[0:10] )

    cv2.imwrite('./Images/' + str(ts)[0:10]  + '/pic_' + str(val) + '.png', swatch)


    #for k in range(0, background_width - SWATCH_NUM_PIXELS_WIDTH):
    # TEMPORARY: ARBITRARY MIDDLE OF IMAGE 
    for k in range(50, 53):    
        prev_im_rep = background.copy()

        prev_im_rep[background_height/2 - SWATCH_NUM_PIXELS_HEIGHT/2:
                background_height/2 - SWATCH_NUM_PIXELS_HEIGHT/2 + SWATCH_NUM_PIXELS_HEIGHT, 
                k:SWATCH_NUM_PIXELS_WIDTH+k] = swatch

        k = k + DELTA
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

    return xFlowTotal/k - BASELINE#, yFlowTotal/k)

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
            #cv2.line(im_rep, midpoint, flowpoint, (0,255,0),1)

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

    global val 
    val = val + 1
    # print each Swatch
    if (val == 1):
        os.makedirs('./Images/' + str(ts)[0:10] )

    cv2.imwrite('./Images/' + str(ts)[0:10]  + '/pic_' + str(val) + '.png', swatch)

        # ZOOM
    #----------------------------------------------------------------------
    # zoom the image a pixel in size

    ## ARE THESE INDEXED CORRECTLY??? 
    # square so doesn't matter for now
    i_x = 200
    i_y = 200

    dim = (i_x, i_y)

    swatch1 = cv2.resize(swatch, dim, interpolation = cv2.INTER_AREA)


    left = center[0]-swatch1.shape[0]/2
    right = center[0]+swatch1.shape[0]/2
    top = center[1]-swatch1.shape[1]/2
    bottom = center[1]+swatch1.shape[1]/2

    prev_im_rep[left:right, top:bottom] = swatch1


    i_x = i_x + 20   #increment 2 pixels so the canvas grows a pixel around each frame
    i_y = i_y + 20 
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

    zoom_val = getZoomOpticalFlow(prv, nxt)
    return zoom_val


def calculatedRotAvg(individual):
    return 0

# Returns the longitudinal OF, vertical OF, for fitness calculations
def evalMax(individual):
    if (option == 0):
        h = calculatedTranslationalAvg_refactored(individual)
    elif (option == 1):
        h = calculatedZoomAvg(individual)
    else:
        h = calculatedRotAvg(individual)
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
    random.seed(64)
    
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
        algorithms.eaMuPlusLambda(pop, toolbox, mu=3, lambda_=POPSIZE, cxpb=0.25, mutpb=0.5, ngen=NUM_GENS, stats=stats, halloffame=hof, verbose=True)
        #algorithms.eaSimple(pop, toolbox, cxpb=0.25, mutpb=0.5, ngen=NUM_GENS, stats=stats, halloffame=hof, verbose=True)
    finally: 
        #1 == 1
        PIC_new = numpy.zeros((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
        for i in range(0, SWATCH_NUM_PIXELS_HEIGHT):
            for j in range (0, SWATCH_NUM_PIXELS_WIDTH):
                PIC_new[i][j] = numpy.array([hof[0][SWATCH_NUM_PIXELS_HEIGHT * i + j],hof[0][SWATCH_NUM_PIXELS_HEIGHT * i + j], hof[0][SWATCH_NUM_PIXELS_HEIGHT * i + j]])

        cv2.imwrite('./Images/' + str(ts)[0:10]  + '/pic_FINAL.png', PIC_new)
        
    return pop, stats, hof

if __name__ == "__main__":
    main(); 
