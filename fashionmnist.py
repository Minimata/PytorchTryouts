"""
Some Fashion MNIST training session
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)


TRAIN_SET = torchvision.datasets.FashionMNIST(
    root='./data/fashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

TRAIN_LOADER = DataLoader(TRAIN_SET, batch_size=10)


class Network(nn.Module):
    """
    Our neural network class
    """
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.dense1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.dense2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # First convolution
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Second convolution
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # third layer
        t = t.reshape(-1, 12*4*4)
        t = self.dense1(t)
        t = F.relu(t)

        # forth layer
        t = self.dense2(t)
        t = F.relu(t)

        # fifth and output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)

        return t



if __name__ == "__main__":
    network = Network()
    with torch.no_grad():
        sample = next(iter(TRAIN_SET))
        image, label = sample
        pred = network(image.unsqueeze(0))
        print(pred.argmax(dim=1).item(), label)
