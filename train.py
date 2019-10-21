from data import voc, bdd
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import argparse
from bdd_dataset import BDDDataset

dataset_root = '/home/krishna/datasets/'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser(
	description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='Pretrained base model')
parser.add_argument('--batch_size', default=40, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--use_cuda', default=False, action="store_true")
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available() and args.use_cuda:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	print('Using CUDA')
else:
	print('Not using CUDA')

if not os.path.exists(args.save_folder):
	os.mkdir(args.save_folder)

num_epochs = 20


def detection_collate(batch):
	"""Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
	targets = []
	imgs = []
	for sample in batch:
		imgs.append(sample[0])
		targets.append(torch.FloatTensor(sample[1]))
	return torch.stack(imgs, 0), targets


def train():
	# cfg = voc
	cfg = bdd
	device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
	dataset = BDDDataset(root=dataset_root, img_size=(cfg['min_dim'], cfg['min_dim']), train=True, config=None)
	# dataset = VOCDetection(root=dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))

	ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
	net = ssd_net

	if args.resume:
		print('Resuming training, loading {}...'.format(args.resume))
		ssd_net.load_weights(args.resume)
	else:
		vgg_weights = torch.load(args.save_folder + args.basenet)
		print('Loading base network...')
		ssd_net.vgg.load_state_dict(vgg_weights)

	net = net.to(device)

	if not args.resume:
		print('Initializing weights...')
		# initialize newly added layers' weights with xavier method
		ssd_net.extras.apply(weights_init)
		ssd_net.loc.apply(weights_init)
		ssd_net.conf.apply(weights_init)

	optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.99, 0.999), weight_decay=5e-4)
	criterion = MultiBoxLoss(cfg['num_classes'], 0.5, 3, device)

	net.train()
	loc_loss = 0
	conf_loss = 0
	print('Loading the dataset...')
	print(len(dataset))
	epoch_size = len(dataset) // args.batch_size
	print('Training SSD on:', dataset.name)
	print('Using the specified args:')
	print(args)

	step_index = 0

	data_loader = data.DataLoader(dataset, args.batch_size, num_workers=4, shuffle=True, collate_fn=detection_collate,
	                              pin_memory=True)
	# create batch iterator
	for epoch in range(args.epochs):
		for iteration, (images, targets) in enumerate(data_loader):
			# load train data
			images = images.to(device)
			targets = [torch.FloatTensor(ann).to(device) for ann in targets]

			# forward
			t0 = time.time()
			out = net(images)
			# backprop
			optimizer.zero_grad()
			loss_l, loss_c = criterion(out, targets)
			loss = loss_l + loss_c
			loss.backward()
			optimizer.step()
			t1 = time.time()
			loc_loss += loss_l.data.item()
			conf_loss += loss_c.data.item()

			if iteration % 50 == 0:
				print('timer: %.4f sec.' % (t1 - t0))
				print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')

			# if iteration % 1000 == 0:
			# 	for param_group in optimizer.param_groups:
			# 		param_group['lr'] = param_group['lr'] * 0.1
		if epoch != 0 and epoch % 5 == 0:
			print('Saving state, epoch:', epoch)
			torch.save(ssd_net.state_dict(), 'weights/ssd300_' + dataset.name + repr(epoch) + '.pth')


def xavier(param):
	init.xavier_uniform_(param)


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		xavier(m.weight.data)
		m.bias.data.zero_()

if __name__ == '__main__':
	train()
