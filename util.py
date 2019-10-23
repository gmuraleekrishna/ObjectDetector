from __future__ import division

import torch
from torchvision.transforms import ToPILImage, transforms
from utils.augmentations import ToAbsoluteCoords
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageDraw, JpegImagePlugin


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


class LetterBox(object):
	def __init__(self, shape=(416, 416)):
		self.shape = shape

	def __call__(self, image, labels):
		img_w, img_h = image.size[0], image.size[1]
		w, h = self.shape
		new_w = int(img_w * min(w / img_w, h / img_h))
		new_h = int(img_h * min(w / img_w, h / img_h))
		resized_image = image.resize((new_w, new_h), resample=Image.BICUBIC)

		h_scale = new_h / img_h
		w_scale = new_w / img_w
		bboxes = np.array(labels, dtype=np.float32)
		h_shift = (h - new_h) // 2
		w_shift = (w - new_w) // 2

		# Scaling Bounding box coordinates and shifting in accordance with gray padding
		bboxes[:, 0] = bboxes[:, 0] * w_scale + w_shift
		bboxes[:, 1] = bboxes[:, 1] * h_scale + h_shift
		bboxes[:, 2] = bboxes[:, 2] * w_scale + w_shift
		bboxes[:, 3] = bboxes[:, 3] * h_scale + h_shift

		canvas = np.full((h, w, 3), 128, dtype=np.uint8)
		canvas[h_shift: h_shift + new_h, w_shift:w_shift + new_w, :] = resized_image
		canvas = Image.fromarray(canvas)
		return canvas, bboxes

def plot_boxes(image, targets):
	class_names = (
		"bike",
		"bus",
		"car",
		"motor",
		"person",
		"rider",
		"traffic light",
		"traffic sign",
		"train",
		"truck"
	)

	if torch.is_tensor(targets):
		targets = torch.Tensor.cpu(targets).detach().numpy()
	else:
		targets = np.array(targets)
	targets = targets.squeeze()
	image = ToPILImage()(image)
	image, targets = ToAbsoluteCoords()(image, boxes=targets)
	targets = targets.astype('int32')
	draw = ImageDraw.Draw(image)

	colors = [(0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 100, 100), (100, 0, 100),
	          (100, 100, 0), (0, 0, 100), (0, 255, 0), (255, 165, 0), (255, 255, 0)]
	# font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
	count = 0
	print('Labels:', targets)
	for result in targets:
		result = result.astype('int32')
		color = colors[result[-1]]
		fcolor = tuple(255 - i for i in color)
		draw.rectangle(list(result[0:4]), outline=color)
		text = "{0}".format(class_names[result[-1]])
		text_size = draw.textsize(text, direction='ltr')
		c1 = tuple(result[0:2])
		c2 = (c1[0] + text_size[0], c1[1] + text_size[1])
		draw.rectangle([c1, c2], color, color)
		draw.text(c1, text, fcolor)
		count += 1
	print('Labels found:', count)
	image.show()

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
