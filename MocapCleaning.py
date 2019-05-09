import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import c3d

import numpy

torch.set_printoptions(linewidth=120)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class BodyMocapCNN(nn.Module):
    def __init__(self, num_markers):
        super(BodyMocapCNN, self).__init__()
        self.num_markers = num_markers
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.dense_input = nn.Linear(168, 1024)
        self.dense_middle = nn.Linear(1024, 1024)
        self.dense_output = nn.Linear(1024, 168)

        self.dense_layers = nn.Sequential(
            self.dense_input,
            nn.ReLU(),
            # self.dense_middle,
            # nn.ReLU(),
            # self.dense_middle,
            # nn.ReLU(),
            # self.dense_middle,
            # nn.ReLU(),
            # self.dense_middle,
            # nn.ReLU(),
            self.dense_output,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = x.reshape(out.size(0), -1)
        out = self.dense_layers(out)
        return out.reshape(out.size(0), 3, 56)


def train(epochs, model, loss_func, train_loaders, valid_loaders, opt):
    final_mean_distance = -1
    final_std = 0
    mean_distance_func = lambda out, target: (target - out).abs().mean().item()
    std_func = lambda out, target: (target - out).std().item()

    for epoch in range(epochs):
        final_mean_distance = 0
        num_valid = 0

        model.train()
        for i, frame in enumerate(zip(*(train_loaders.values()))):
            mean = 0
            std = 0
            num_frames = 0
            for actor_frame in frame:
                unclean = actor_frame[0]
                clean = actor_frame[1]

                opt.zero_grad()
                out = model(unclean)
                loss = loss_func(out, clean)
                loss.backward()
                opt.step()

                mean += mean_distance_func(out, clean)
                std += std_func(out, clean)
                num_frames += unclean.shape[0]

            # Print loss and accuracy
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Mean distance: {:.2f}mm, Standard deviation: {:.2f}mm'
                      .format(epoch + 1, epochs,
                              i + 1, len(next(iter(train_loaders.values()))),
                              mean / num_frames, std / num_frames))

        model.eval()
        with torch.no_grad():
            for i, frame in enumerate(zip(*(valid_loaders.values()))):
                for actor_frame in frame:
                    unclean = actor_frame[0]
                    clean = actor_frame[1]

                    out = model(unclean)
                    final_mean_distance += loss_func(out, clean).item()
                    final_std += std_func(out, clean)
                    num_valid += unclean.shape[0]

        final_mean_distance /= num_valid
        final_std /= num_valid
        print('Epoch [{}/{}], Validation Mean Distance : {:.2f}mm, Validation Standard Deviation : {:.2f}mm'
              .format(epoch + 1, epochs, final_mean_distance, final_std))
    return final_mean_distance, final_std


def load_mocap_data(marker_numbers, batch_size, train_data_proportion=0., pure_validation=False):
    unclean, clean = {}, {}
    for name in marker_numbers:
        unclean[name] = []
        clean[name] = []

    def reshape_to_conv_frame(t):
        return t.transpose(0, 1)

    with open('./data/mocap/20170825_021_uncleaned.c3d', 'rb') as uncleaned, \
            open('./data/mocap/20170825_021_cleaned.c3d', 'rb') as cleaned:
        uncleaned_reader = c3d.Reader(uncleaned)
        cleaned_reader = c3d.Reader(cleaned)
        for unclean_frame, clean_frame in zip(uncleaned_reader.read_frames(), cleaned_reader.read_frames()):
            idx = 0
            for name, num_mark in marker_numbers.items():
                base_unclean_frame = torch.tensor(unclean_frame[1], dtype=torch.float32)
                unclean[name].append(reshape_to_conv_frame(base_unclean_frame[idx:idx + num_mark, 0:3]))
                base_clean_frame = torch.tensor(clean_frame[1], dtype=torch.float32)
                clean[name].append(reshape_to_conv_frame(base_clean_frame[idx:idx + num_mark, 0:3]))
                idx += num_mark

    dataloaders = {}
    for name in marker_numbers:
        unclean[name] = torch.stack(unclean[name]).to(DEVICE)
        clean[name] = torch.stack(clean[name]).to(DEVICE)
        # print("{} \t- \tunclean: {}, \tclean: {}".format(name, unclean[name].shape, clean[name].shape))

        if pure_validation:
            valid_ds = MocapDataset(unclean[name], clean[name])
            dataloaders[name] = DataLoader(valid_ds, batch_size=batch_size)
        else:
            training_data_length = int(len(unclean[name]) * train_data_proportion)
            train_ds = MocapDataset(unclean[name][0:training_data_length], clean[name][0:training_data_length])
            valid_ds = MocapDataset(unclean[name][training_data_length:len(unclean[name])], clean[name][training_data_length:len(unclean[name])])
            dataloaders[name] = {
                "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                "valid": DataLoader(valid_ds, batch_size=batch_size)
            }

    return dataloaders


class MocapDataset(Dataset):
    """Mocap dataset"""

    def __init__(self, unclean_frames, clean_frames, transform=None):
        super(MocapDataset, self).__init__()
        self.unclean_frames = unclean_frames
        self.clean_frames = clean_frames
        self.transform = transform

    def __len__(self):
        return len(self.unclean_frames)

    def __getitem__(self, index):
        sample = [self.unclean_frames[index], self.clean_frames[index]]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def write_c3d_with_model(model, dataloader, file):
    writer = c3d.Writer()
    model.eval()
    with torch.no_grad():
        for i, actor in enumerate(dataloader):
            frame = actor[0]  # take the unclean data frame only to give to the model
            out = model(frame)
            outs = out.reshape(-1, 3, 56).transpose(1, 2).unbind()
            null_tensor = torch.zeros(56, 2, dtype=torch.float32, device=DEVICE)
            np_outs = [torch.cat((t, null_tensor), dim=1).cpu().numpy() for t in outs]
            for t in np_outs:
                writer.add_frames([t, numpy.zeros((0, 1))])
            if (i + 1) % 100 == 0:
                print('Processed frames [{}/{}]'.format(i + 1, len(dataloader)))

    with open(file, 'wb') as f:
        print("Writing data to {}...".format(file))
        writer.write(f)

if __name__ == "__main__":
    # training setup
    batch_size = 32  # batch size
    loss_func = lambda out, target: (target - out).abs().mean()
    learning_rate = 0.00001  # learning rate
    epochs = 100  # how many epochs to train for
    # marker_numbers = {
    #     'JRO': 56,
    #     'AMA': 56,
    #     'prop1': 5,
    #     'prop2': 3
    # }
    marker_numbers = {
        'JRO': 56,
        'AMA': 56
    }

    mocap_dl = load_mocap_data(marker_numbers, batch_size, train_data_proportion=0.9)
    train_loaders, valid_loaders = {}, {}
    for name, loaders in mocap_dl.items():
        train_loaders[name] = loaders['train']
        valid_loaders[name] = loaders['valid']

    model = BodyMocapCNN(marker_numbers['JRO']).to(DEVICE)
    adam = optim.Adam(model.parameters(), lr=learning_rate)


    train(epochs, model, loss_func, train_loaders, valid_loaders, adam)
    torch.save(model, './models/bodymocapCNN.pt')

    dataloaders = load_mocap_data(marker_numbers, batch_size, pure_validation=True)
    test_model = torch.load('./models/bodymocapCNN.pt', map_location=DEVICE)
    jro_dl, ama_dl = [dl for name, dl in dataloaders.items()]
    write_c3d_with_model(test_model, jro_dl, './data/mocap/predicted/20170825_021_predicted_JRO.c3d')
    write_c3d_with_model(test_model, ama_dl, './data/mocap/predicted/20170825_021_predicted_AMA.c3d')
