# Black square move CV test
import numpy as np
import cv2
from pylab import imshow, show

source = 0
cap = cv2.VideoCapture(source)
size = (320,240)
# cap.set(cv2.CAP_PROP_FPS,5) #apparently these constants are defined...
cap.set(3, size[0]) # video width setting
cap.set(4, size[1]) # video height setting
cap.set(5,2) # framerate setting

while True:

	# draw the video to screen
	img = cap.read()
	cv2.imshow('FramerateDemo', img[1])

	# escape key to quit (insert image of fire escape here)
	ch = 0xFF & cv2.waitKey(5)
	if ch == 27:
		break

cv2.destroyAllWindows()