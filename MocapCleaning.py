import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

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
    marker_numbers = {
        'JCO': 56,
        'AMA': 56,
        'prop1': 5,
        'prop2': 3
    }
    unclean_frames = clean_frames = {
        'JCO': [],
        'AMA': [],
        'prop1': [],
        'prop2': []
    }

    with open('./data/mocap/20170825_021_uncleaned.c3d', 'rb') as unclean, \
         open('./data/mocap/20170825_021_cleaned.c3d', 'rb') as clean:
        unclean_reader = c3d.Reader(unclean)
        clean_reader = c3d.Reader(clean)
        for unclean_frame, clean_frame in zip(unclean_reader.read_frames(), clean_reader.read_frames()):
            idx = 0
            for name, num_mark in marker_numbers.items():
                unclean_frames[name].append(torch.tensor(unclean_frame[1])[idx:idx+num_mark, 0:3])
                clean_frames[name].append(torch.tensor(clean_frame[1])[idx:idx+num_mark, 0:3])
                idx += num_mark

    for name in unclean_frames:
        print(len(unclean_frames[name]), len(clean_frames[name]))
        unclean_frames[name] = torch.stack(unclean_frames[name])
        print(unclean_frames[name].shape)
        # clean_frames[name] = torch.stack(clean_frames[name])
        print("{}\t->\tunclean: {}\tclean: {}".format(name, unclean_frames[name].shape, clean_frames[name][0].shape))


    # training setup
    batch_size = 50  # batch size
    loss_func = F.cross_entropy  # loss function
    learning_rate = 0.001  # learning rate
    epochs = 10  # how many epochs to train for
    num_of_classes = 10
  
    network = OtherNetwork(10).to(DEVICE)
    adam = optim.Adam(network.parameters(), lr=learning_rate)

    train_dl, validation_dl = load_data_set(batch_size)
    fit(epochs, network, loss_func, train_dl, validation_dl, adam)
