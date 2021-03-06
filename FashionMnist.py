import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

torch.set_printoptions(linewidth=120)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.dense1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.dense2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.dense1(t.reshape(-1, 12 * 4 * 4)))
        t = F.relu(self.dense2(t))
        t = self.out(t)

        return t

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
    # training setup
    batch_size = 50  # batch size
    loss_func = F.cross_entropy  # loss function
    learning_rate = 0.001  # learning rate
    epochs = 10  # how many epochs to train for
    num_of_classes = 10

    # network = Network().to(DEVICE)
    network = OtherNetwork(10).to(DEVICE)
    adam = optim.Adam(network.parameters(), lr=learning_rate)
    SGD = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(SGD, patience=1, threshold=0.001, verbose=True)

    train_dl, validation_dl = load_data_set(batch_size)
    adam_acc = fit(epochs, network, loss_func, train_dl, validation_dl, adam)
    sgd_acc = fit(epochs, network, loss_func, train_dl, validation_dl, SGD, scheduler)

    print("SGD Accuracy: {} <-> Adam Accuracy: {}".format(sgd_acc, adam_acc))
