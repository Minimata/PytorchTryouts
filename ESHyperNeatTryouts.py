import os

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import gym
import neat

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.recurrent_net import RecurrentNet
from pytorch_neat.es_hyperneat import ESNetwork
from pytorch_neat.substrate import Substrate
from pytorch_neat.cppn import create_cppn

torch.set_printoptions(linewidth=120)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class MyCNN(nn.Module):
    def __init__(self, num_of_class):
        super(MyCNN, self).__init__()
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
        val_loss = 0
        for i, (images, labels) in enumerate(train_dl):
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
                predicted = torch.argmax(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                final_acc = 100 * correct / total

            print('Epoch [{}/{}], Test Accuracy : {:.2f}%'.format(epoch + 1, epochs, final_acc))
    return final_acc


max_env_steps = 200

def make_env():
    env = gym.make("CartPole-v0")
    return env

def make_net(genome, config, bs):
    #start by setting up a substrate for this bad cartpole boi
    params = {"initial_depth": 2,
              "max_depth": 4,
              "variance_threshold": 0.00013,
              "band_threshold": 0.00013,
              "iteration_level": 3,
              "division_threshold": 0.00013,
              "max_weight": 3.0,
              "activation": "tanh"}
    input_cords = []
    output_cords = [(0.0, -1.0, 0.0)]
    sign = 1
    for i in range(4):
        input_cords.append((0.0 - i/10*sign, 1.0, 0.0))
        sign *= -1
    leaf_names = []
    for i in range(3):
        leaf_names.append('leaf_one_'+str(i))
        leaf_names.append('leaf_two_'+str(i))

    [cppn] = create_cppn(genome, config, leaf_names, ['cppn_out'])
    # print(cppn)
    print(leaf_names)
    print(config.genome_config.input_keys)
    net_builder = ESNetwork(Substrate(input_cords, output_cords), cppn, params)
    net = net_builder.create_phenotype_network_nd()
    return net

def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return outputs[:, 0] > 0.5


if __name__ == "__main__":
    # training setup
    batch_size = 50
    loss_func = F.cross_entropy
    learning_rate = 0.001
    epochs = 10
    num_of_classes = 10
    n_generations = 100

    train_dl, validation_dl = load_data_set(batch_size)

    ### Traditional CNN
    # network = BodyMocap(10).to(DEVICE)
    # adam = optim.Adam(network.parameters(), lr=learning_rate)
    # fit(epochs, network, loss_func, train_dl, validation_dl, adam)

    ### ES-HyperNEAT
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MultiEnvEvaluator(
        make_net, 
        activate_net, 
        batch_size=batch_size,
        make_env=make_env, 
        max_env_steps=max_env_steps
    )

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("neat.log", evaluator.eval_genome)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)
