<<<<<<< Updated upstream
import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn import Module, Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, Sequential

class Model(Module)
	def __init__(self):
		self.block1 = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3 ,stride=1, padding=1),
			BatchNorm2d(16),
			LeakyReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)
		self.block2 = Sequential(
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(32),
			LeakyReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)
		self.block3 = Sequential(
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(64),
			LeakyReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)
		self.block4 = Sequential(
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(128),
			LeakyReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)

		self.block5 = Sequential(
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(256),
			LeakyReLU(),
			MaxPool2d(kernel_size=2, stride=2)
		)

		self.block6 = Sequential(
			Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(512),
			LeakyReLU(),
			MaxPool2d(kernel_size=2, stride=1)
		)
		self.block7 = Sequential(
			Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(1024),
			LeakyReLU(),
		)
		self.block8 = Sequential(
			Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=1),
			BatchNorm2d(256),
			LeakyReLU(),
		)
		self.block9 = Sequential(
			Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(512),
			LeakyReLU(),
		)
		self.block9 = Conv2d(in_channels=512, out_channels=255, kernel_size=1, stride=1, padding=1)
