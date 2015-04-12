import argparse
import cv2 as opencv
import json
from helpers import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("imgs", help="Path to the image(s) in which to find faces.", nargs="+")
parser.add_argument("--output", help="Path to json output.", default="STDOUT")
parser.add_argument("--face_cascade", help="Path to cascade file.", default="data/haarcascade_frontalface_alt.xml")
parser.add_argument("--eye_cascade", help="Path to eye cascade file.", default="data/haarcascade_eye.xml")
args = parser.parse_args()
	
face_cascade = opencv.CascadeClassifier(args.face_cascade)
if(face_cascade.empty()):
	raise Exception("Face cascade not found: {}".format(args.face_cascade))

eye_cascade = opencv.CascadeClassifier(args.eye_cascade)
if(eye_cascade.empty()):
	raise Exception("Eye cascade not found: {}".format(args.eye_cascade))

results = {}

for image in args.imgs:
	color = opencv.imread(image)
	gray = opencv.cvtColor(color, opencv.COLOR_BGR2GRAY)

	img_area = color.shape[0] * color.shape[1]
	
	result = {"faces" : []}
	
	print(image)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print(faces)

	for(x,y,w,h) in faces:
		face_area = w * h
		result["faces"].append({"x": x, "y": y, "w":w, "h":h, "scale":(face_area/float(img_area))})
		face = gray[y:(y+h), x:(x+w)]
		eyes = eye_cascade.detectMultiScale(face)
		
		if(len(eyes) == 2):
			if(eyes[0][0] < eyes[1][0]):
				eye_left=[eyes[0][0] + eyes[0][2]/2.0, eyes[0][1] + eyes[0][3]/2.0]
				eye_right=[eyes[1][0] + eyes[1][2]/2.0, eyes[1][1] + eyes[1][3]/2.0]
			else:
				eye_left=[eyes[1][0] + eyes[1][2]/2.0, eyes[1][1] + eyes[1][3]/2.0]
				eye_right=[eyes[0][0] + eyes[0][2]/2.0, eyes[0][1] + eyes[0][3]/2.0]

			result["faces"][-1]["eyes"] = {"left": {"x" : eye_left[0]+x, "y": eye_left[1]+y}, "right": {"x" : eye_right[0]+x, "y": eye_right[1]+y}}

	results[image] = result

json_string = json.dumps(results, cls=NumpyAwareJSONEncoder)

if args.output == "STDOUT":
	print json_string
else:
	f = open(args.output, "w")
	f.write(json_string)
	f.close()
