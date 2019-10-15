import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sampler import ImbalancedDatasetSampler
from focalloss import *

from sklearn.model_selection import train_test_split
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
np.random.seed(0) #numpy
torch.backends.cudnn.deterministic=True # cudnn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1)
        self.conv3 = nn.Conv2d(128, 64, 3, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(13 * 13 * 64, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 13 * 13 * 64)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    data_transform = transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(root='train_sets', transform=data_transform)
    train_set = torch.utils.data.DataLoader(train_data, batch_size=30, shuffle=True, num_workers=30)

    test_data = datasets.ImageFolder(root='val_sets', transform=data_transform)
    test_set = torch.utils.data.DataLoader(test_data, batch_size=30, shuffle=False, num_workers=30)

    print(len(train_set), len(test_set))
    classes = ('0', '1', '2', '3', '4')



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Net()

    criterion = nn.FocalLoss(gamma=0)(x,l)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    train_loss_list = []
    val_loss_list = []


    ax = []
    ay = []
    by = []
    cy = []
    dy = []
    for epoch in range(50):  # loop over the dataset multiple times
        ax.append(epoch)
        running_loss = 0.0
        val_loss = 0.0
        for i, data in enumerate(train_set, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 98 == 97:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 98))
                train_loss_list.append(running_loss / 98)
                running_loss = 0.0
        ay.append(train_loss_list[-1])
        plt.plot(ax, ay, 'r-')

        with torch.no_grad():
            for j, data in enumerate(test_set, 0):
                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = net(inputs).to(device)
                loss = criterion(outputs, labels).to(device)

                # print statistics
                val_loss += loss.item()
                if j % 25 == 24:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, j + 1, val_loss / 25))
                    val_loss_list.append(val_loss / 25)
                    val_loss = 0.0
        by.append(val_loss_list[-1])
        plt.plot(ax, by, 'g-')
        plt.show()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in train_set:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs).to(device)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            cy.append(correct / total)
        plt.figure(2)
        plt.plot(ax, cy, 'r-')

        correct = 0
        total = 0
        with torch.no_grad():
            confusion_matrix = torch.zeros(5, 5)
            for data in test_set:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs).to(device)
                _, predicted = torch.max(outputs.data, 1)
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        dy.append(correct / total)
        plt.plot(ax, dy, 'g-')
        plt.show()
        print(confusion_matrix)
        print(confusion_matrix.diag() / confusion_matrix.sum(1))

    print('finished')

    # dataiter = iter(test_set)
    # images, labels = dataiter.next()

    # print images
    # imshow(torchvision.utils.make_grid(images))