import json
from torch.utils.data import Dataset
import torch
import os
from augmentations import transform
from PIL import Image
from utils import letterbox #, print_boxes_original
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
        """
        :param index:
        :return: (image, labels): labels are in form [x1, y1, x2, y2, class_idx]
        """
        annotation = self.annotations[index]
        image_file_name = annotation["name"]
        image = Image.open(os.path.join(self.root, self.image_folder, image_file_name))
        self.size = image.size
        targets = np.zeros(5*50)
        labels = np.array(annotation['labels'])
        image, labels[:, 1:5] = letterbox(image, labels[:,1:5], self.img_size)
        # image = transform(image, bboxes, classes, config=self.config)

        # Changing format to (x, y, w, h) from (x0, y0, x1, y1)
        cx = (labels[:, 1] + labels[:, 3])/2
        cy = (labels[:, 2] + labels[:, 4])/2
        w = labels[:, 3] - labels[:, 1]
        h = labels[:, 4] - labels[:, 2]
        labels[:, 1] = cx
        labels[:, 2] = cy
        labels[:, 3] = w
        labels[:, 4] = h

        labels = labels.reshape(-1)
        if labels.shape[0] > 5*50:
            targets = labels[0:50*5]
        elif labels.shape[0] > 0:
            targets[0:labels.shape[0]] = labels

        image = functional.to_tensor(image)
        targets = torch.from_numpy(targets.astype('float32'))

        return image, targets


if __name__ == "__main__":
#    root = "/home/krishna/datasets/"
    root = "E:/ANU Study Stuff/Semester 3/Advanced Topics in Mechatronics/Project/BDD100K/"
    dataset = BDDDataset(root)
    _ , labels = dataset[105]
    # print_boxes_original(labels, dataset, 105)
#    # image = cv2.imread(os.path.join(root, 'bdd100k/images/train/', annotation['name']), cv2.IMREAD_UNCHANGED)
#    image = ToPILImage()(image)
#
#    for label in labels.numpy():
#        # class_ = BDDDabtaset.class_names.index(class_)
#        c1 = tuple(label[0:2])
#        c2 = tuple(label[2:4])
#        image = cv2.rectangle(np.asarray(image), c1, c2, (0, 255, 0), 2)
#        label = "{0}".format(dataset.class_names[label[-1]])
#        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
#        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
#        cv2.rectangle(image, c1, c2, (0, 255, 0), -1)
#        cv2.putText(image, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
#
#    cv2.imshow("image", image)
#    while True:
#        key = cv2.waitKey(1)
#        if key == 27:
#            break
