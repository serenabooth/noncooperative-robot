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
import cv2
import time
import os

from pylab import imshow, show, ion

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from _functools import partial

#ion()
SWATCH_NUM_PIXELS_WIDTH = 100
SWATCH_NUM_PIXELS_HEIGHT = 100
POPULATION = 40
NGEN = 10
BLACK = 0
WHITE = 255
background_width = 320
background_height = 240
DELTA = 1
POPSIZE = 10
NUM_GENS = 100 
ts = time.time() 
val = 0

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

# PIC is the 100x100 swatch, background is the 320x240 image
#PIC = genRandomImage(SWATCH_NUM_PIXELS_HEIGHT, SWATCH_NUM_PIXELS_WIDTH)
PIC = numpy.zeros((SWATCH_NUM_PIXELS_HEIGHT, SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
background = numpy.zeros((background_height,background_width,3), numpy.uint8)
#background = genRandomImage(background_height, background_width)

# insert randomness through pixel generation
def gen_random_pixels(): 
    color = random.randint(0,1) # returns 0 or 1 for b or w 
    #color = WHITE
    if (color == 1):
        color = WHITE
    return tuple([random.randint(0, SWATCH_NUM_PIXELS_HEIGHT - 1), 
        random.randint(0, SWATCH_NUM_PIXELS_WIDTH - 1), color])

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox() 
history = tools.History()

toolbox.register("attr", gen_random_pixels)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=SWATCH_NUM_PIXELS_WIDTH * SWATCH_NUM_PIXELS_HEIGHT)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def calculatedTranslationalAvg(individual):

    swatch = numpy.zeros((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
    for i in range(0,SWATCH_NUM_PIXELS_HEIGHT):
        for j in range(0,SWATCH_NUM_PIXELS_WIDTH):
            swatch[i][j] = individual[i][j]



    #for i in range(0, len(individual)):
    #    swatch[individual[i][0]][individual[i][1]] = numpy.array([individual[i][2], individual[i][2], individual[i][2]])

    xFlowTotal = 0 
    yFlowTotal = 0    

    #imshow(swatch)
    #show()

    global val 
    val = val + 1

    if (val == 1):
        os.makedirs('./Images/' + str(ts)[0:10] )

    cv2.imwrite('./Images/' + str(ts)[0:10]  + '/pic_' + str(val) + '.png', swatch)


    #for k in range(0, background_width - SWATCH_NUM_PIXELS_WIDTH):
    # TEMPORARY: ARBITRARY MIDDLE OF IMAGE 
    for k in range(50, 53):    
        prev_im_rep = background.copy()
        prev_im_rep[70:170, k:100+k] = swatch

        k = k + DELTA
        im_rep_next = background.copy()
        im_rep_next[70:170, k:100+k] = swatch 

        # make CV happy with grayscale images for previous and next frames
        prv = cv2.cvtColor(prev_im_rep, cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(im_rep_next, cv2.COLOR_BGR2GRAY) 

        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prv, nxt, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # sum our optical flow and then get averages for x and y direction
        total = cv2.sumElems(flow)
        xFlowTotal += total[0] / (background.shape[0]*background.shape[1])
        yFlowTotal += total[1] / (background.shape[0]*background.shape[1])

    return -1 * xFlowTotal/k#, yFlowTotal/k)


# Returns the longitudinal OF, vertical OF, for fitness calculations
def evalMax(individual):

    ind = PIC.copy()
    for tri in individual:
        ind[tri[0]][tri[1]] = numpy.array([tri[2], tri[2], tri[2]])

    h = calculatedTranslationalAvg(ind)
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
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)
# Decorate the variation operators
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

def main():
    random.seed(64)

    #imshow(PIC)
    #show()
    
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
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof)
    finally: 
        PIC_new = PIC.copy()
        for tri in hof[0]:
            PIC_new[tri[0]][tri[1]] = numpy.array([tri[2], tri[2], tri[2]])
        cv2.imwrite('./Images/' + str(ts)[0:10]  + '/pic_FINAL.png', PIC_new)




    #for i in range(1, len(history.genealogy_history) + 1):
    #    PIC_new = numpy.zeros((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)

    #    for tri in history.genealogy_history[i]:
    #        PIC_new[tri[0]][tri[1]] = numpy.array([tri[2], tri[2], tri[2]])

    #return pop, stats, hof

if __name__ == "__main__":
    main(); 
