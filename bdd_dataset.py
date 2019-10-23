import json
import random

from torch.utils.data import Dataset
import torch
import os
from PIL import Image, ImageDraw, ImageEnhance
from util import LetterBox, plot_boxes
import numpy as np
from torchvision.transforms import functional, ToPILImage, transforms
import cv2
from utils.augmentations import ToAbsoluteCoords


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for box in target:
            box[0] = box[0] / self.shape[0]
            box[1] = box[1] / self.shape[1]
            box[2] = box[2] / self.shape[0]
            box[3] = box[3] / self.shape[1]
            res.append(box)
        return image, res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class RandomMirror(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image, labels):
        _, width = self.shape
        if random.randint(0, 1):
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
            labels[:, 0::3] = width - labels[:, 2::-2]
        return image, labels


class RandomContrast(object):
    def __init__(self, lower=0.1, upper=1.0):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, labels):
        if random.randint(0, 1):
            alpha = random.uniform(self.lower, self.upper)
            ImageEnhance.Contrast(image).enhance(alpha)
        return image, labels


class BDDDataset(Dataset):
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

    def __init__(self, root, img_size=(416, 416), train=True, config=None):
        if config is None:
            self.config = {'expand': False, 'flip': False, 'normalise': False, 'resize': False, 'distort': False,
                           'crop': False}
        else:
            self.config = config
        self.name = 'BDD100k'
        self.root = root
        self.img_size = img_size
        self.config = config
        self.is_train = train
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
        image, labels, h, w = self.pull_item(index)
        if self.is_train:
            image, labels = RandomMirror(shape=self.img_size)(image, labels)
            image, labels = RandomContrast()(image, labels)
        image, labels = LetterBox(shape=self.img_size)(image, labels)
        if self.is_train:
            image, labels = AnnotationTransform(shape=self.img_size)(image, labels)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
        # image = transform(image, bboxes, classes, config=self.config)
        return image, labels, index

    def pull_item(self, index):
        annotation = self.annotations[index]
        image_file_name = annotation["name"]
        image = Image.open(os.path.join(self.root, self.image_folder, image_file_name))
        labels = annotation['labels']

        if self.is_train:
            return image, labels, image.size[0], image.size[1]
        else:
            image, labels = LetterBox(shape=self.img_size)(image, labels)
            return image, labels, image.size[0], image.size[1]

    def pull_image(self, index):
        annotation = self.annotations[index]
        image_file_name = annotation["name"]
        image = Image.open(os.path.join(self.root, self.image_folder, image_file_name))
        image, _ = LetterBox(shape=self.img_size)(image, [(0, 0, 0, 0, 0)])
        return transforms.ToTensor()(image)

    def pull_anno(self, index):
        annotation = self.annotations[index]
        labels = annotation['labels']
        image_file_name = annotation["name"]
        return image_file_name, labels


if __name__ == "__main__":
    root = "/home/krishna/datasets/"
    dataset = BDDDataset(root)
    image, classes, bboxes = dataset.__getitem__(105)
    image = ToPILImage()(image)

    for class_, bbox in zip(classes.numpy(), bboxes.numpy()):
        c1 = tuple(bbox[0:2])
        c2 = tuple(bbox[2:4])
        image = cv2.rectangle(np.asarray(image), c1, c2, (0, 255, 0), 2)
        label = "{0}".format(dataset.class_names[label[-1]])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(image, c1, c2, (0, 255, 0), -1)
        cv2.putText(image, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)

    cv2.imshow("image", image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
