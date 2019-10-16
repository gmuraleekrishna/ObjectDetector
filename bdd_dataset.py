import json
from torch.utils.data import Dataset
import os
from Pillow import Image


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

	def __init__(self, root, train=True, transform=False):
		self.root = root
		if train:
			self.image_folder = 'bdd100k/images/100k/train'
			# TODO: find label json folder path
			annotations_file = ''
		else:
			self.image_folder = 'bdd100k/images/100k/val'
			# TODO: find label folder path
			annotations_file = ''
		self._transform = transform
		self.annotations = {}
		with open(annotations_file, 'rb') as annotations_json:
			self.annotations = json.loads(annotations_json)

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, index):
		annotation = self.annotations[index]
		image_file_name = annotation["name"]
		image = Image.open(os.path.join(self.image_folder, image_file_name))
		classes = []
		bboxes = []
		for label in annotation['labels']:
			classes.append(BDDDataset.class_names.index(label["category"]))
			bbox = [
				int(label["box2d"]["x1"]), int(label["box2d"]["y1"]),
				int(label["box2d"]["x2"]), int(label["box2d"]["y2"])
			]
			bboxes.append(bbox)

		return image, classes, bboxes
