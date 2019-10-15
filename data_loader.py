from kitti_dataset import KittiDataset
from utils import transform
from torch import 

def load_data():
	train_loader = KittiDataset('./dataset', 'training', transform=transform)