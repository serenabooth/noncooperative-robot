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

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from _functools import partial

def genRandomImage():
    im_rep = np.zeros((NUM_PIXELS,NUM_PIXELS,3), int)

    i = 0 
    while (i < 0.5 * NUM_PIXELS * NUM_PIXELS):
        x = random.randint(0,NUM_PIXELS -1)
        y = random.randint(0,NUM_PIXELS -1)
        if (im_rep[x][y][0] == 0):
            im_rep[x][y] = np.array([1,1,1])
            i = i + 1

    return im_rep

NUM_PIXELS = 100 
POPULATION = 40
NGEN = 100
PIC = genRandomImage()

def gen_one_random_pixel(): 
    color = random.randint(0,1) # returns 0 or 1 for b or w 
    return tuple([random.randint(0, NUM_PIXELS - 1), random.randint(0, NUM_PIXELS - 1), color])

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", gen_one_random_pixel)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def expected(individual):
    return 1

def calculated(individual):
    return 0

# SERENA: evalPattern should compute invoke OpenCV optic flow call; should be difference between real motion and calculated OF? Should penalize (badly) for all single color image. We're maximizing the difference between the calculated movement and computed OF diff. 
def evalOneMax(individual):
    newPIC = PIC
    for tri in individual:
        print tri
        newPIC[tri[0], tri[1]] = tri[2]
    return expected(newPIC) - calculated(newPIC), 
    #return individual,

# SERENA: as we use numpy.ndarray representation, this should work in a similar way for cxTwoPatterns
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
    
    
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    #random.seed(64)
    
    pop = toolbox.population(n=300)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats,
                        halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    main(); 
