from __future__ import division
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from bdd_dataset import BDDDataset

# SELECT NETWORK

# network = "yolov3"
network = "yolov3-tiny"


# network = "yolov2"
# network = "yolov2-tiny"
# network = "yolov1"
# network = "yolov1-tiny"       ## Pre-trained weights not found

def arg_parse():
	"""
    Parse arguements to the detect module

    """

	parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

	parser.add_argument("--root", dest='data_path', help=
						"Root Directory containing complete dataset",
	                    default="E:\ANU Study Stuff\Semester 3\Advanced Topics in Mechatronics\Project\BDD100K",
	                    type=str)
	parser.add_argument("--det", dest='det', help=
						"Image / Directory to store detections to",
	                    default="det", type=str)
	parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
	parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
	parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
	parser.add_argument("--cfg", dest='cfgfile', help=
						"Config file",
	                    default="cfg/" + network + ".cfg", type=str)
	parser.add_argument("--weights", dest='weightsfile', help=
						"weightsfile",
	                    default="weights/" + network + ".weights", type=str)
	parser.add_argument("--reso", dest='reso', help=
						"Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
	                    default="416", type=str)

	return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso  # DOUBT!!
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
	model.cuda()

# Set the model in evaluation mode
model.eval()

train_data = BDDDataset(args.data_path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
num_classes = 10

if not os.path.exists(args.det):
	os.makedirs(args.det)

write = 0
start_det_loop = time.time()
for i, batch in enumerate(train_data):
	# load the image
	start = time.time()
	image, classes, bboxes = batch
	if CUDA:
		image = image.cuda()

	prediction = model(Variable(image, volatile=True), CUDA)

	prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

	end = time.time()

	if type(prediction) == int:

		# for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
		# 	im_id = i * batch_size + im_num
		# 	print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
		# 	print("{0:20s} {1:s}".format("Objects Detected:", ""))
		# 	print("----------------------------------------------------------")
		continue

	prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in training list

	if not write:  # If we have't initialised output
		output = prediction
		write = 1
	else:
		output = torch.cat((output, prediction))

	for im_num, image in enumerate(image):
		im_id = i * batch_size + im_num
		objs = [train_data.class_names[int(x[-1])] for x in output if int(x[0]) == im_id]
		print("{0:20s} predicted in {1:6.3f} seconds".format(train_data.annotations[im_id]['name'], (end - start) / batch_size))
		print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
		print("----------------------------------------------------------")

	if CUDA:
		torch.cuda.synchronize()

try:
    output
except NameError:
    print ("No detections were made")
    exit()