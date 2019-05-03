from pathlib import Path
import requests

import pickle
import gzip
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import torchvision
from torchvision import transforms

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

class MnistDataLoader:
    def __init__(self, dl):
        self.dl = dl

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (MnistDataLoader.preprocess(*b))

    @staticmethod
    def preprocess(input_batch, output_batch):
        return input_batch.view(-1, 1, 28, 28).to(DEV), output_batch.to(DEV)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def load_data_set(batch_size=64):
    root = './data/fashionMNIST'
    train_set = torchvision.datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validation_set = torchvision.datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    validation_loader = DataLoader(validation_set, batch_size=batch_size)

    return MnistDataLoader(train_loader), MnistDataLoader(validation_loader)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    # else:
    #     print("hello")

    return loss.item(), len(xb)

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def main():
    # Fetch MNIST data if not already present on filesystem
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    # load MNIST dataset
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

        # training setup
        batch_size = 64  # batch size
        loss_func = F.cross_entropy  # loss function
        learning_rate = 0.1  # learning rate
        # learning_rate = 0.5  # learning rate
        epochs = 10  # how many epochs to train for

        def get_model_CNN():
            model = MnistCNN()
            return model, optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        def get_model():
            model = MnistLogistic()
            return model, optim.SGD(model.parameters(), lr=learning_rate)

        def get_model_custom():
            model = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                Lambda(lambda x: x.view(x.size(0), -1)),
            )
            model.to(DEV)
            return model, optim.Adam(model.parameters(), lr=learning_rate)

        # build tensor
        x_train, y_train, x_valid, y_valid = map(
            torch.tensor, (x_train, y_train, x_valid, y_valid)
        )

        train_ds = TensorDataset(x_train, y_train)
        valid_ds = TensorDataset(x_valid, y_valid)
        other_train_dl, other_valid_dl = get_data(train_ds, valid_ds, bs=batch_size)

        train_dl, valid_dl = load_data_set(batch_size)

        t, v = next(iter(train_dl))
        print("Train DL: ", t.shape, v.shape)
        t, v = next(iter(other_train_dl))
        print("Other DL: ", t.shape, v.shape)
        t, v = next(iter(valid_dl))
        print("Valid DL: ", t.shape, v.shape)
        t, v = next(iter(other_valid_dl))
        print("Other DL: ", t.shape, v.shape)

        train_dl = MnistDataLoader(train_dl)
        valid_dl = MnistDataLoader(valid_dl)

        model, opt = get_model_custom()
        fit(epochs, model, loss_func, opt, train_dl, valid_dl)

if __name__ == '__main__':
    main()
