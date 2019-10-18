import json
from torch.utils.data import Dataset
import torch
import os
from augmentations import transform
from PIL import Image
from util import letterbox
import numpy as np
from torchvision.transforms import functional, ToPILImage
import cv2


class BDDDataset(Dataset):
	class_names = [
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
	]

	def __init__(self, root, img_size=(416, 416), train=True, config=None):
		if config is None:
			self.config = {'expand': False, 'flip': False, 'normalise': False, 'resize': False, 'distort': False,
			               'crop': False}
		else:
			self.config = config

		self.root = root
		self.img_size = img_size
		self.config = config
		if train:
			self.image_folder = 'bdd100k/images/train'
			self.annotations_file = 'bdd100k/converted_labels/bdd100k_labels_images_train.json'
		else:
			self.image_folder = 'bdd100k/images/val'
			self.annotations_file = 'bdd100k/converted_labels/bdd100k_labels_images_val.json'
		self.annotations = {}
		with open(os.path.join(self.root, self.annotations_file), 'r') as annotation_json:
			self.annotations = json.load(annotation_json)

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, index):
		annotation = self.annotations[index]
		image_file_name = annotation["name"]
		image = Image.open(os.path.join(self.root, self.image_folder, image_file_name))
		bboxes = annotation['bboxes']
		classes = list(map(lambda class_name: BDDDataset.class_names.index(class_name), annotation['classes']))
		image, bboxes = letterbox(image, bboxes, self.img_size)
		# image = transform(image, bboxes, classes, config=self.config)
		image = functional.to_tensor(image)
		bboxes = torch.IntTensor(bboxes)
		classes = torch.IntTensor(classes)
		return image, classes, bboxes


if __name__ == "__main__":
	root = "E:\ANU Study Stuff\Semester 3\Advanced Topics in Mechatronics\Project\BDD100K"
	dataset = BDDDataset(root)
	image, classes, bboxes = dataset.__getitem__(0)
	# image = cv2.imread(os.path.join(root, 'bdd100k/images/train/', annotation['name']), cv2.IMREAD_UNCHANGED)
	image = ToPILImage()(image)
	for class_, bbox in zip(classes.numpy(), bboxes.numpy()):
		# class_ = BDDDataset.class_names.index(class_)
		image = cv2.rectangle(np.asarray(image), tuple(bbox[0:2]), tuple(bbox[2:4]), (0, 255, 0), 1)
	cv2.imshow("image", image)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break
