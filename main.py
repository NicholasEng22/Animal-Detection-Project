import cv2
import imutils
import numpy as np

#from darkflow.net.build import TFNet
import cv2

from imutils.video import VideoStream
from imutils import resize

import cv2 as cv
import matplotlib.pyplot as plt

import os
import threading
import time

import subprocess

# Video Capture 
# capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture("Videos/natureClip.mp4")
#capture = cv2.VideoCapture("Videos/Caterpiller.mp4")
capture = cv2.VideoCapture("Videos/BirdDemo.mp4")
#capture = cv2.VideoCapture("Videos/dog.mp4")

# Numpy random number things (just for getting random colors.)
from numpy.random import default_rng
rng = default_rng(42)

#get frames per second
fps = capture.get(cv2.CAP_PROP_FPS) #(cv2.CV_CAP_PROP_FPS

# History, Threshold, DetectShadows 
# fgbg = cv2.createBackgroundSubtractorMOG2(50, 200, True)
fgbg = cv2.createBackgroundSubtractorMOG2(300, 100, True)

#Intialize Average
avg = None

# Keeps track of what frame we're on
frameCount = 0

# Get "class" names from COCO training data set and get random colors
# associated with each.
classes = open("coco.names").read().strip().split("\n")
colors = rng.integers(0, 255, size=(len(classes), 3), dtype="uint8")

# Now load the net object.
net = cv.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# Get output layer names for later.
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#Function for drawing boxes around objects
def drawBoxes(frame, outputs):
	# Now draw some boxes around objects with high confidence.
	boxes = []
	confidences = []
	classIDs = []
	h, w = frame.shape[:2]

	for output in outputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > 0.2:
				box = detection[:4] * np.array([w, h, w, h])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				box = [x, y, int(width), int(height)]
				boxes.append(box)
				confidences.append(float(confidence))
				classIDs.append(classID)

	indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	if len(indices) > 0:
		for i in indices.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in colors[classIDs[i]]]
			cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
			cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
	return frame

while(1):
	# Return Value and the current frame
	ret, frame = capture.read()

	#  Check if a current frame actually exist
	if not ret:
		break

	# Resize the frame
	resizedFrame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

	# The net is expecting a blob.
	# Set the input and get the output.
	print(resizedFrame.shape)
	blob = cv.dnn.blobFromImage(resizedFrame, 1/255.0, resizedFrame.shape[0:2], swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(ln)

	#Call function to draw boxes
	resizedFrame = drawBoxes(resizedFrame, outputs)

	#blurs and converts image to grayscale
	gray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the average frame is None, initialize it
	if avg is None:
		avg = gray.copy().astype("float")
		#capture.truncate(0)
		continue

	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

	# threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, 7.5, 255,
		cv2.THRESH_BINARY)[1] #5
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 2500: #threshold number for contours
			continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(resizedFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Animal"

	frameCount += 1
	time = float(frameCount)/fps

	# Get the foreground mask
	#fgmask = fgbg.apply(resizedFrame)

	# Count all the non zero pixels within the mask
	#count = np.count_nonzero(fgmask)

	#print('Frame: %d, FPS: %d, Pixel Count: %d' % (frameCount, fps, count))

	# Determine how many pixels do you want to detect to be considered "movement"
	# if (frameCount > 1 and cou`nt > 5000):

	# if (frameCount > 1 and count > 17000): #experiment with the thresh hold number
	# 	print('Animal detected: %f' % time)
	# 	cv2.putText(resizedFrame, 'Animal detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
	# 	#Output timestamp

	cv2.imshow('Frame', resizedFrame)
	# cv2.imshow('Mask', fgmask)

	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break

capture.release()
cv2.destroyAllWindows()
