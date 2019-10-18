from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
	batch_size = prediction.size(0)
	stride = inp_dim // prediction.size(2)
	grid_size = inp_dim // stride
	bbox_attrs = 5 + num_classes
	num_anchors = len(anchors)

	prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
	prediction = prediction.transpose(1, 2).contiguous()
	prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

	anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

	# Sigmoid the  centre_X, centre_Y. and object confidencce
	prediction[:, :, 0] = F.sigmoid(prediction[:, :, 0])
	prediction[:, :, 1] = F.sigmoid(prediction[:, :, 1])
	prediction[:, :, 4] = F.sigmoid(prediction[:, :, 4])

	# Add the center offsets
	grid = np.arange(grid_size)
	a, b = np.meshgrid(grid, grid)

	x_offset = torch.FloatTensor(a).view(-1, 1)
	y_offset = torch.FloatTensor(b).view(-1, 1)

	if CUDA:
		x_offset = x_offset.cuda()
		y_offset = y_offset.cuda()

	x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

	prediction[:, :, :2] += x_y_offset

	# log space transform height and the width
	anchors = torch.FloatTensor(anchors)

	if CUDA:
		anchors = anchors.cuda()

	anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
	prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

	prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
	prediction[:, :, :4] *= stride

	return prediction


def unique(tensor):
	tensor_np = tensor.cpu().numpy()
	unique_np = np.unique(tensor_np)
	unique_tensor = torch.from_numpy(unique_np)

	tensor_res = tensor.new(unique_tensor.shape)
	tensor_res.copy_(unique_tensor)
	return tensor_res


def bbox_iou(box1, box2):
	"""
	Returns the IoU of two bounding boxes


	"""
	# Get the coordinates of bounding boxes
	b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
	b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

	# get the corrdinates of the intersection rectangle
	inter_rect_x1 = torch.max(b1_x1, b2_x1)
	inter_rect_y1 = torch.max(b1_y1, b2_y1)
	inter_rect_x2 = torch.min(b1_x2, b2_x2)
	inter_rect_y2 = torch.min(b1_y2, b2_y2)

	# Intersection area
	inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
	                                                                                 min=0)

	# Union Area
	b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
	b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

	iou = inter_area / (b1_area + b2_area - inter_area)

	return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
	conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
	prediction = prediction * conf_mask

	box_corner = prediction.new(prediction.shape)
	box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
	box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
	box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
	box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
	prediction[:, :, :4] = box_corner[:, :, :4]

	batch_size = prediction.size(0)

	write = False

	for ind in range(batch_size):
		image_pred = prediction[ind]  # image Tensor

		max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
		max_conf = max_conf.float().unsqueeze(1)
		max_conf_score = max_conf_score.float().unsqueeze(1)
		seq = (image_pred[:, :5], max_conf, max_conf_score)
		image_pred = torch.cat(seq, 1)

		non_zero_ind = (torch.nonzero(image_pred[:, 4]))
		try:
			image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
		except:
			continue

		# For PyTorch 0.4 compatibility
		# Since the above code with not raise exception for no detection
		# as scalars are supported in PyTorch 0.4
		if image_pred_.shape[0] == 0:
			continue
		# Get the various classes detected in the image
		img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index

		for cls in img_classes:
			# get the detections with one particular class
			cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
			class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
			image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

			# sort the detections such that the entry with the maximum objectness
			# confidence is at the top
			conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
			image_pred_class = image_pred_class[conf_sort_index]
			idx = image_pred_class.size(0)  # Number of detections

			for i in range(idx):
				# Get the IOUs of all boxes that come after the one we are looking at
				# in the loop
				try:
					ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
				except ValueError:
					break

				except IndexError:
					break

				# Zero out all the detections that have IoU > threshold
				iou_mask = (ious < nms_conf).float().unsqueeze(1)
				image_pred_class[i + 1:] *= iou_mask

				# Remove the non-zero entries
				non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
				image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
				batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
				# Repeat the batch_id for as many detections of the class cls in the image
				seq = batch_ind, image_pred_class

				if not write:
					output = torch.cat(seq, 1)
					write = True
				else:
					out = torch.cat(seq, 1)
					output = torch.cat((output, out))
	try:
		return output
	except:
		return 0


def letterbox(img, bboxes, inp_dim):
	"""resize image and bounding boxes with unchanged aspect ratio using padding"""

	img_w, img_h = img.size[0], img.size[1]
	w, h = inp_dim
	new_w = int(img_w * min(w / img_w, h / img_h))
	new_h = int(img_h * min(w / img_w, h / img_h))
	resized_image = img.resize((new_w, new_h), resample=Image.BICUBIC)

	h_scale = new_h / img_h
	w_scale = new_w / img_w
	bboxes = np.array(bboxes, dtype=np.long)

	# Scaling Bounding box coordinates
	bboxes[:, 0] = bboxes[:, 0] * w_scale
	bboxes[:, 1] = bboxes[:, 1] * h_scale
	bboxes[:, 2] = bboxes[:, 2] * w_scale
	bboxes[:, 3] = bboxes[:, 3] * h_scale

	h_shift = (h - new_h) // 2
	w_shift = (w - new_w) // 2
	canvas = np.full((inp_dim[1], inp_dim[0], 3), 128, dtype=np.uint8)
	canvas[h_shift: h_shift + new_h, w_shift:w_shift + new_w, :] = resized_image
	canvas = Image.fromarray(canvas)

	# Shifting Bounding Box coordinates in accordance with gray padding
	bboxes[:, 0] = bboxes[:, 0] + w_shift
	bboxes[:, 1] = bboxes[:, 1] + h_shift
	bboxes[:, 2] = bboxes[:, 2] + w_shift
	bboxes[:, 3] = bboxes[:, 3] + h_shift

	return canvas, bboxes

def load_weights(self, weightfile):
	#Open the weights file
    fp = open(weightfile, "rb")

    #The first 5 values are header information 
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number 
    # 4,5. Images seen by the network (during training)
    header = np.fromfile(fp, dtype = np.int32, count = 5)
    self.header = torch.from_numpy(header)
    self.seen = self.header[3]

	weights = np.fromfile(fp, dtype = np.float32)
	ptr = 0
    for i in range(len(self.module_list)):
        module_type = self.blocks[i + 1]["type"]

        #If module_type is convolutional load weights
        #Otherwise ignore.
		if module_type == "convolutional":
            model = self.module_list[i]
            try:
                batch_normalize = int(self.blocks[i+1]["batch_normalize"])
            except:
                batch_normalize = 0

            conv = model[0]
			if (batch_normalize):
				bn = model[1]

				#Get the number of weights of Batch Norm Layer
				num_bn_biases = bn.bias.numel()

				#Load the weights
				bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
				ptr += num_bn_biases

				bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
				ptr  += num_bn_biases

				bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
				ptr  += num_bn_biases

				bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
				ptr  += num_bn_biases

				#Cast the loaded weights into dims of model weights. 
				bn_biases = bn_biases.view_as(bn.bias.data)
				bn_weights = bn_weights.view_as(bn.weight.data)
				bn_running_mean = bn_running_mean.view_as(bn.running_mean)
				bn_running_var = bn_running_var.view_as(bn.running_var)

				#Copy the data to model
				bn.bias.data.copy_(bn_biases)
				bn.weight.data.copy_(bn_weights)
				bn.running_mean.copy_(bn_running_mean)
				bn.running_var.copy_(bn_running_var)
			else:
				#Number of biases
				num_biases = conv.bias.numel()

				#Load the weights
				conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
				ptr = ptr + num_biases

				#reshape the loaded weights according to the dims of the model weights
				conv_biases = conv_biases.view_as(conv.bias.data)

				#Finally copy the data
				conv.bias.data.copy_(conv_biases)

				#Let us load the weights for the Convolutional layers
				num_weights = conv.weight.numel()

				#Do the same as above for weights
				conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
				ptr = ptr + num_weights

				conv_weights = conv_weights.view_as(conv.weight.data)
				conv.weight.data.copy_(conv_weights)
		