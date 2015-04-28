# Pixel combinations for nxn images
# by Jonathan Bobrow
# April 27, 2015
#
# based on iterative funcion found here:
# https://docs.python.org/2/library/itertools.html#itertools.combinations

# size sets a nxn image with combinations of all possible pixel arrangements
# ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡
# ¡¡¡¡¡¡careful!!!!
# !!!!!!!!!!!!!!!!!
# number of files = 2^(size*size)

import numpy
import numpy as np
import cv2
import time
import itertools

size = 2			# set size here
length = size*size	# create our string for naming and combinations (genotype...)


ts = time.time() 

def kbits(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = ['0'] * n
        for bit in bits:
            s[bit] = '1'
        result.append(''.join(s))

        img = numpy.zeros((size,size,3), numpy.uint8)
        for i in range(0, size):
        	for j in range(0, size):
        		img[i][j] = 255*int(s[i*size+j])
        # print img
        # print s
        label = ""
        for i in range(0, size*size):
        	label += s[i]
        cv2.imwrite('./Images/' + label + '.png', img)
    return result


for i in range(0,length+1):
	comb = kbits(length, i)
	#print comb