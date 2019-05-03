import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from ezc3d import c3d as ezc3d
import c3d

torch.set_printoptions(linewidth=120)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class OtherNetwork(nn.Module):
    def __init__(self, num_of_class):
        super(OtherNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_of_class)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


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

    return train_loader, validation_loader


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def fit(epochs, model, loss_func, train_dl, valid_dl, opt, scheduler=None):
    final_acc = -1
    for epoch in range(epochs):
        model.train()
        i = 0
        val_loss = 0
        for images, labels in train_dl:
            i += 1
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            opt.zero_grad()
            out = model(images)
            loss = loss_func(out, labels)
            val_loss += loss.item()
            loss.backward()
            opt.step()

            acc = accuracy(out, labels)

            # Print loss and accuracy
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, epochs, i + 1, len(train_dl), loss.item(), acc * 100))

        if scheduler is not None:
            loss = val_loss / len(train_dl)
            # print(loss)
            scheduler.step(loss)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in valid_dl:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                out = model(images)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                final_acc = 100 * correct / total

            print('Epoch [{}/{}], Test Accuracy : {:.2f}%'.format(epoch + 1, epochs, final_acc))
    return final_acc


if __name__ == "__main__":
    file = './data/mocap/20170825_021_uncleaned.c3d'

    points = []
    with open(file, 'rb') as handle:
        reader = c3d.Reader(handle)
        print(reader.header)
        print(reader.point_labels)
        for p in reader.read_frames():
            marker_frame = torch.tensor(p[1])[:, 0:3]
            points.append(marker_frame)

    points = torch.stack(points)
    print(points.shape)

    # mocap_uncleaned = ezc3d('./data/mocap/BDE_FACS_0954.c3d')
    # print(mocap_uncleaned['parameters']['POINT']['USED']['value'][0])  # Print the number of points used
    # point_data = mocap_uncleaned['data']['points']
    # analog_data = mocap_uncleaned['data']['analogs']


    # training setup
    # batch_size = 50  # batch size
    # loss_func = F.cross_entropy  # loss function
    # learning_rate = 0.001  # learning rate
    # epochs = 10  # how many epochs to train for
    # num_of_classes = 10
    #
    # network = OtherNetwork(10).to(DEVICE)
    # adam = optim.Adam(network.parameters(), lr=learning_rate)
    #
    # train_dl, validation_dl = load_data_set(batch_size)
    # fit(epochs, network, loss_func, train_dl, validation_dl, adam)
