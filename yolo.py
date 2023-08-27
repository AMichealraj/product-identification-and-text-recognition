import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image

FLAGS = [0]

if __name__ == '__main__':



	labels = open('yolov3-coco/coco-labels').read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(os.path.abspath("yolov3-coco/yolov3.cfg"),os.path.abspath("yolov3-coco/yolov3.weights"))

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]



	count = 0

	vid = cv.VideoCapture(0)
	while True:
		_, frame = vid.read()
		height, width = frame.shape[:2]

		if count == 0:
			frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS=False)
			count += 1
		else:
			frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS, boxes,
																	confidences, classids, idxs, infer=False)
			count = (count + 1) % 6

			#text = "{}: {:4f}".format(labels[classids[i]], confidences[i])

		cv.imshow('webcam', frame)

		if cv.waitKey(1) & 0xFF == ord('q'):
			break
	vid.release()
	cv.destroyAllWindows()


