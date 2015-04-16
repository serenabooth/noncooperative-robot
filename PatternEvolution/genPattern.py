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
PIC = genRandomImage(SWATCH_NUM_PIXELS_HEIGHT, SWATCH_NUM_PIXELS_WIDTH)
background = numpy.zeros((background_height,background_width,3), numpy.uint8)

# insert randomness through pixel generation
def gen_one_random_pixel(): 
    color = random.randint(0,1) # returns 0 or 1 for b or w 
    if (color == 1):
        color = WHITE
    return tuple([random.randint(0, SWATCH_NUM_PIXELS_HEIGHT - 1), random.randint(0, SWATCH_NUM_PIXELS_WIDTH - 1), color])

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr", gen_one_random_pixel)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compute swatch from individual. Place swatch on background. Move translationally. Compute OF. 
def calculated(individual):
    swatch = numpy.empty((SWATCH_NUM_PIXELS_HEIGHT,SWATCH_NUM_PIXELS_WIDTH,3), numpy.uint8)
    for i in range(0,SWATCH_NUM_PIXELS_HEIGHT):
        for j in range(0,SWATCH_NUM_PIXELS_WIDTH):
            swatch[i][j] = individual[i][j]

    k = 0
    prev_im_rep = background.copy()
    prev_im_rep[70:170, k:100+k] = swatch

    k = k + 10
    im_rep_next = background.copy()
    im_rep_next[70:170, k:100+k] = swatch 

    # make CV happy with grayscale images for previous and next frames
    prv = cv2.cvtColor(prev_im_rep, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(im_rep_next, cv2.COLOR_BGR2GRAY) 

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prv, nxt, 0.5, 4, 8, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # sum our optical flow and then get averages for x and y direction
    total = cv2.sumElems(flow)
    xFlowAve = total[0] / (background.shape[0]*background.shape[1])
    yFlowAve = total[1] / (background.shape[0]*background.shape[1])

    return xFlowAve

# Returns the longitudinal OF over the number of pixels moved. This fitness is minimized
def evalMax(individual, pixels):
    for tri in pixels:
        individual[tri[0]][tri[1]] = numpy.array([tri[2], tri[2], tri[2]])
    imshow(individual)
    show()
    #return (expected(individual) - calculated(individual)),
    return calculated(individual),

# cross over function -- provided by DEAP
def cxTwoPointCopy(ind1, ind2):
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
    
    
toolbox.register("evaluate", partial(evalMax, PIC))
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    
    pop = toolbox.population(n=10)
    
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
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    main(); 
