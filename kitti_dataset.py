from torch.utils.data import Dataset
import os
import torch
import csv


class KittiDataset(Dataset):
	class_names = [
		'Car',
		'Van',
		'Truck',
		'Pedestrian',
		'Person_sitting',
		'Cyclist',
		'Tram',
		'Misc',
		'Void'
	]

	def __init__(self, root, split='training', transform=False):
		self.root = root
		self.split = split
		self._transform = transform
		self.images_file_names = os.listdir(os.path.join(self.root, split, 'image'))

	def __len__(self):
		return len(self.images_file_names)

	def __getitem__(self, index):
		classes = []
		boxes = []
		image_file_name = self.images_file_names[index]
		label_file_name = image_file_name[:-4] + '.txt'
		with open(label_file_name, 'r') as csv_fl:
			csv_reader = csv.reader(csv_fl, delimiter=' ')
			for annotation in csv_reader:
				classes.append(KittiDataset.class_names.index(annotation[0]))
				boxes.append(annotation[4:9])

		self._transform (image, classes, )

		return img, lbl