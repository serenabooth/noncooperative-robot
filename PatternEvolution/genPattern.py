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

from pylab import imshow, show

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from _functools import partial

def genRandomImage():
    im_rep = numpy.zeros((NUM_PIXELS,NUM_PIXELS,3), int)

    i = 0 
    while (i < 0.5 * NUM_PIXELS * NUM_PIXELS):
        x = random.randint(0,NUM_PIXELS -1)
        y = random.randint(0,NUM_PIXELS -1)
        if (im_rep[x][y][0] == 0):
            im_rep[x][y] = numpy.array([1,1,1])
            i = i + 1

    return im_rep

NUM_PIXELS = 100 
POPULATION = 40
NGEN = 10
PIC = genRandomImage()

def gen_one_random_pixel(): 
    color = random.randint(0,1) # returns 0 or 1 for b or w 
    return tuple([random.randint(0, NUM_PIXELS - 1), random.randint(0, NUM_PIXELS - 1), color])

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# TODO: ERROR WHEN USING THE gen_one_random_pixel approach! 
toolbox.register("attr", gen_one_random_pixel)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def expected(individual):
    return 0

def calculated(individual):
    return 1

# SERENA: evalPattern should compute invoke OpenCV optic flow call; should be difference between real motion and calculated OF? Should penalize (badly) for all single color image. We're maximizing the difference between the calculated movement and computed OF diff. 
def evalMax(individual, triangles):
    for tri in triangles:
        individual[tri[0]][tri[1]] = numpy.array([tri[2], tri[2], tri[2]])
    imshow(individual)
    show()
    return (expected(individual) - calculated(individual)),
    #return individual,


def evalOneMax(individual):
    print individual
    return sum(individual),

#def evalOneMax(individual):
#    return sum(individual),

# SERENA: as we use numpy.ndarray representation, this should work in a similar way for cxTwoPatterns
def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
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
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    #random.seed(64)
    
    pop = toolbox.population(n=1)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=numpy.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    #stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    main(); 
