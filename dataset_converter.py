import json
import os

from bdd_dataset import BDDDataset

new_annotation = []
dataset_type = ['train', 'val'][0]
dataset_folder = 'E:\ANU Study Stuff\Semester 3\Advanced Topics in Mechatronics\Project\BDD100K'

if __name__ == "__main__":
	output_dir = os.path.join(dataset_folder, 'bdd100k/converted_labels/')
	input_dir = os.path.join(dataset_folder, 'bdd100k/labels/')
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	with open(os.path.join(input_dir, f'bdd100k_labels_images_{dataset_type}.json'), 'r') as json_fl:
		jsn = json.load(json_fl)
		for annotation in jsn:
			file_name = annotation["name"]
			classes = []
			bboxes = []
			for label in annotation["labels"]:
				if 'box2d' not in label or 'category' not in label:
					continue
				if label['category'] not in BDDDataset.class_names:
					continue
				classes.append(label['category'])
				bbox = [int(label['box2d']["x1"]), int(label['box2d']["y1"]), int(label['box2d']["x2"]), int(label['box2d']["y2"])]
				bboxes.append(bbox)
			new_annotation.append({
				"name": file_name,
				"classes": classes,
				"bboxes": bboxes
			})

	with open(os.path.join(output_dir, f'bdd100k_labels_images_{dataset_type}.json'), 'w') as output_jsn:
		json.dump(new_annotation, output_jsn)