from bdd_dataset import BDDDataset
from utils import transform


def load_data():
	train_loader = BDDDataset('./dataset', 'training', transform=transform)