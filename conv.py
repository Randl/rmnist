"""conv.py
~~~~~~~~~~

A simple convolutional network for the RMNIST data sets.  Adapted from
code in the pytorch documentation:
http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

"""

from __future__ import print_function

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm, trange

import data_loader

# Configuration
n = 1  # use RMNIST/n
expanded = True  # Whether or not to use expanded RMNIST training data
if n == 0: epochs = 100
if n == 1: epochs = 500
if n == 5: epochs = 400
if n == 10: epochs = 200
# We decrease the learning rate by 20% every 10 epochs
if n == 0:
    lr = 0.01
else:
    lr = 0.1
batch_size = 10
momentum = 0.0
mean_data_init = 0.1
sd_data_init = 0.25
seed = 1
torch.manual_seed(seed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean_data_init,), (sd_data_init,))
])


class RMNIST(Dataset):
    def __init__(self, n=0, train=True, transform=None, expanded=False):
        self.n = n
        self.transform = transform
        td, vd, ts = data_loader.load_data(n, expanded=expanded)
        if train:
            self.data = td
        else:
            self.data = vd

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        data = self.data[0][idx]
        img = (data * 256)
        img = img.reshape(28, 28)
        img = Image.fromarray(np.uint8(img))
        if self.transform: img = self.transform(img)
        label = self.data[1][idx]
        return img, label


train_dataset = RMNIST(n, train=True, transform=transform, expanded=expanded)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
training_data = list(train_loader)

validation_dataset = RMNIST(n, train=False, transform=transform, expanded=expanded)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=100, shuffle=True)
validation_data = list(validation_loader)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(320, 50),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 320)
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()


def train(epoch):
    optimizer = optim.SGD(model.parameters(), lr=lr * ((0.8) ** (epoch / 10 + 1)), momentum=momentum)
    # optimizer = optim.Adam(model.parameters(), lr=lr*0.1, weight_decay=1e-4)
    model.train()
    for batch_idx, (data, target) in enumerate(training_data):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def accuracy():
    if (n != 0) and (epoch % 10 != 0):
        return
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in validation_data:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    validation_loss /= 10000
    tqdm.write('Validation set: average loss: {:.4f}, accuracy: {}/{} ({:.1f}%)\n'.format(
        validation_loss, correct, 10000, 100. * correct / 10000))


for epoch in trange(1, epochs + 1):
    train(epoch)
    accuracy()
