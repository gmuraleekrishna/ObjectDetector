from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from data import VOC_ROOT, VOC_CLASSES as labelmap
from bdd_dataset import BDDDataset
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, action='store_true',
                    help='Use cuda to train model')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for id in range(num_images):
        print('Testing image {:d}/{:d}....'.format(id+1, num_images))
        img = testset.pull_image(id)
        img_id, annotation = testset.pull_anno(id)
        x = img.unsqueeze(0)

        with open(filename, mode='a') as f:
            f.write(f"\nGROUND TRUTH FOR: {id}\n")
            for box in annotation:
                f.write('['+','.join(str(b) for b in box) + '],\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[2],
                             img.shape[1], img.shape[2]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = BDDDataset.class_names[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write('[' + ','.join(str(b) for b in coords) + ',' + str(i-1) + '],\n')
                j += 1


def test_voc():
    # load net
    num_classes = 11 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = BDDDataset(root='/home/krishna/datasets', img_size=(300, 300), train=False, config=None)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset)


if __name__ == '__main__':
    test_voc()
