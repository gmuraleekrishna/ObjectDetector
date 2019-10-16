import json
from torch.utils.data import Dataset
import os
from PIL import Image


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
			self.image_folder = 'bdd100k/images/train'
			self.annotations_file = 'bdd100k/converted_labels/bdd100k_labels_images_train.json'
		else:
			self.image_folder = 'bdd100k/images/val'
			self.annotations_file = 'bdd100k/converted_labels/bdd100k_labels_images_val.json'
		self._transform = transform
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
		return image, classes, bboxes


if __name__ == "__main__":
	dataset = BDDDataset('/home/krishna/datasets')
	print(dataset.__getitem__(1))