# import mahotas as mh
from pylab import imshow, show
import numpy as np
import random 

NUM_PIXELS = 100 

im_rep = np.zeros((NUM_PIXELS,NUM_PIXELS,3), int)

i = 0 
while (i < 0.5 * NUM_PIXELS * NUM_PIXELS):
    x = random.randint(0,NUM_PIXELS -1)
    y = random.randint(0,NUM_PIXELS -1)
    if (im_rep[x][y][0] == 0):
        im_rep[x][y] = np.array([1,1,1])
        i = i + 1

imshow(im_rep)
show()
