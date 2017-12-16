"""anneal.py
~~~~~~~~~~~~

Do a (modified) simulated anneal to find hyper-parameters for RMNIST.
Also enables the use of an ensemble of multiple neural nets, which
together effectively vote for an answer.

"""

# Standard library
from __future__ import print_function

import math
import random

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

# My library
import data_loader

use_gpu = torch.cuda.is_available()

# Configuration
n = 1  # use RMNIST/n
expanded = True  # Whether or not to use expanded RMNIST training data
if n == 0: epochs = 100
if n == 1: epochs = 500
if n == 5: epochs = 400
if n == 10: epochs = 200
batch_size = 64
momentum = 0.0
mean_data_init = 0.1
sd_data_init = 0.25
seed = 1
torch.manual_seed(seed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean_data_init,), (sd_data_init,))
])

# These are the hyper-parameters that can be annealed.  Note that this
# set could easily be expanded. lr is the learning rate, nk1 is the
# number of kernels in the first layer, and nk2 the number in the
# second layer.
#
# We will use an ensemble of ensemble_size nets.  This shouldn't be
# annealed --- performance will usually get better as we make this
# larger, but it will also extend training time, so the annealing will
# run slower and slower.
params = {"weight_decay": 5e-4 * (10 ** 0.25),
          "lr": 0.1 * (10 ** 0.5),
          "nk1": 20, "nk2": 42, "nlin": 300,
          "ensemble_size": 1}


# Define the annealing moves
def weight_decay_up(params):
    trial = dict(params)
    trial["weight_decay"] *= 10 ** 0.25
    return trial


def weight_decay_down(params):
    trial = dict(params)
    trial["weight_decay"] /= 10 ** 0.25
    return trial


def lr_up(params):
    trial = dict(params)
    trial["lr"] *= 10 ** 0.25
    return trial


def lr_down(params):
    trial = dict(params)
    trial["lr"] /= 10 ** 0.25
    return trial


def k1_up(params):
    trial = dict(params)
    trial["nk1"] += 2
    return trial


def k1_down(params):
    trial = dict(params)
    if trial["nk1"] > 2: trial["nk1"] -= 2
    return trial


def k2_up(params):
    trial = dict(params)
    trial["nk2"] += 2
    return trial


def k2_down(params):
    trial = dict(params)
    if trial["nk2"] > 2: trial["nk2"] -= 2
    return trial



def nlin_up(params):
    trial = dict(params)
    trial["nlin"] = int(1.1*trial["nlin"])
    return trial


def nlin_down(params):
    trial = dict(params)
    if trial["nlin"] > 10: trial["nlin"] = int(0.9*trial["nlin"])
    return trial

moves = [weight_decay_up, weight_decay_down, lr_up, lr_down, k1_up, k1_down, k2_up, k2_down, nlin_up, nlin_down]


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
        return (img, label)


train_dataset = RMNIST(n, train=True, transform=transform, expanded=expanded)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
training_data = list(train_loader)

validation_dataset = RMNIST(n, train=False, transform=transform, expanded=expanded)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=100, shuffle=True)
validation_data = list(validation_loader)

def conv_out_size(in_size, kernel_size, stride = 1, padding =0, dilation=1):
    return int(np.floor((in_size+2*padding-dilation*(kernel_size-1)-1)/stride+1))
def conv_out_ceil_size(in_size, kernel_size, stride = 1, padding =0, dilation=1):
    return int(np.ceil((in_size+2*padding-dilation*(kernel_size-1)-1)/stride+1))

class Net(nn.Module):
    def __init__(self, activation, params):
        super(Net, self).__init__()
        ks1 = 7
        nk1 = params["nk1"]
        ks2 = 4
        nk2 = params["nk2"]
        nlin = params["nlin"]
        self.conv1_out_size = conv_out_size(28, ks1)
        self.mp1_out_size = conv_out_ceil_size(self.conv1_out_size , 3, stride=2)
        self.conv2_out_size = conv_out_size( self.mp1_out_size, ks2)
        self.mp2_out_size = conv_out_ceil_size(self.conv2_out_size, 3, stride=2)
        self.lin = (self.mp2_out_size ** 2) * nk2

        self.activation = activation
        self.conv = nn.Sequential(
            nn.Conv2d(1, nk1, kernel_size=ks1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            self.activation(inplace=True),
            nn.Conv2d(nk1, nk2, kernel_size=ks2),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            self.activation(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.lin, nlin),
            nn.Dropout(),
            self.activation(inplace=True),
            nn.Linear(nlin, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.lin)
        x = self.fc(x)
        return F.log_softmax(x)


def train(epoch, model):
    optimizer = optim.SGD(model.parameters(), lr=params["lr"] * (0.8 ** (epoch / 10 + 1)), momentum=momentum,
                          weight_decay=params["weight_decay"])
    model.train()
    for batch_idx, (data, target) in enumerate(training_data):
        if use_gpu:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def accept(model):
    """Return True if more than 20% of the validation data is being
    correctly classified. Used to avoid including nets which haven't
    learnt anything in the ensemble.

    """

    accuracy = 0
    for data, target in validation_data[:(500 // 100)]:
        if use_gpu:
            data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
        else:
            data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        accuracy += pred.eq(target.data.view_as(pred)).cpu().sum()
    if accuracy < 100:
        return False
    else:
        return True


def ensemble_accuracy(models):
    for model in models:
        model.eval()
    models = [model for model in models if accept(model)]
    print("Number of models used from ensemble: {}".format(len(models)))
    accuracy = 0
    for data, target in validation_data:
        if use_gpu:
            data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
        else:
            data, target = Variable(data, volatile=True), Variable(target)
        outputs = [model(data) for model in models]
        pred = sum(output.data for output in outputs).max(1, keepdim=True)[1]
        accuracy += pred.eq(target.data.view_as(pred)).cpu().sum()
    return accuracy


def run():
    if use_gpu:
        models = [Net(nn.ReLU, params).cuda() for j in range(params["ensemble_size"])]
    else:
        models = [Net(nn.ReLU, params) for j in range(params["ensemble_size"])]
    for j, model in enumerate(models):
        print("Training model: {}".format(j))
        for epoch in range(1, epochs + 1):
            train(epoch, model)
    accuracy = ensemble_accuracy(models)
    print('Validation set ensemble accuracy: {}/{} ({:.0f}%)'.format(
        accuracy, 10000, 100. * accuracy / 10000))
    return accuracy


def hash_dict(d):
    """Construct a hash of the dict d. A problem with this kind of hashing
    is when the values are floats - the imprecision of floating point
    arithmetic mean that values will be regarded as different which
    should really be regarded as the same.  To solve this problem we
    hash to 8 significant digits, by multiplying by 10**8 and then
    rounding to an integer.  It's an imperfect solution, but works
    pretty well in practice.

    """
    l = []
    for k, v in d.items():
        if type(v) == float:
            l.append((k, round(v * (10 ** 8))))
        else:
            l.append((k, v))
    return hash(frozenset(l))


def add_dict_to_cache(cache, d, value):
    cache[hash_dict(d)] = value


def get_value_from_cache(cache, d):
    return cache[hash_dict(d)]


def dict_in_cache(cache, d):
    return hash_dict(d) in cache


energy_scale = 50
cache = {}  # To store accuracies for past hyper-parameter configurations
count = 0
print("\nMove: {}".format(count))
print("Initial params: {}".format(params))
accuracy = run()
best_accuracy = accuracy
best_params = params
add_dict_to_cache(cache, params, accuracy)
keep_going = False  # flag to say whether or not the last move resulted
# in an improvement in accuracy, and we should
# repeat the move.  Not standard in simulated
# annealing.
while True:
    if not keep_going: random_move = random.randint(0, len(moves) - 1)
    count += 1
    print("\nMove: {}".format(count))
    print("Current accuracy: {}".format(accuracy))
    print("Current params: {}".format(params))
    print("Move: {}".format(moves[random_move].__name__))
    trial_params = moves[random_move](params)
    print("Trialling: {}".format(trial_params))
    if dict_in_cache(cache, trial_params):
        print("Retrieving from cache")
        trial_accuracy = get_value_from_cache(cache, trial_params)
        print('Validation set ensemble accuracy: {}/{} ({:.0f}%)'.format(
            trial_accuracy, 10000, 100. * trial_accuracy / 10000))
    else:
        print("Computing from new parameters")
        trial_accuracy = run()
        add_dict_to_cache(cache, trial_params, trial_accuracy)
    keep_going = (trial_accuracy > accuracy)
    if random.random() < math.exp(-(accuracy - trial_accuracy) / energy_scale):
        print("Move accepted")
        params = trial_params
        accuracy = trial_accuracy
    else:
        print("Move not accepted")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
    print("Best accuracy so far: {}".format(best_accuracy))
    print("Best params so far: {}".format(best_params))
