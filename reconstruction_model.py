import argparse
import os
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F


class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()

        k_size = 5

        self.conv1 = nn.Conv2d(3, 32, k_size)
        self.conv2 = nn.Conv2d(32, 64, k_size)
        self.conv3 = nn.Conv2d(64, 128, k_size)
        self.conv4 = nn.Conv2d(128, 10, k_size)

        self.conv5 = nn.ConvTranspose2d(10, 128, k_size)
        self.conv6 = nn.ConvTranspose2d(128, 64, k_size)
        self.conv7 = nn.ConvTranspose2d(64, 32, k_size)
        self.conv8 = nn.ConvTranspose2d(32, 3, k_size)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = self.conv8(out)

        return out


class Skip_ConvAE(nn.Module):
    def __init__(self):
        super(Skip_ConvAE, self).__init__()

        k_size = 5

        self.conv1 = nn.Conv2d(3, 32, k_size)
        self.conv2 = nn.Conv2d(32, 64, k_size)
        self.conv3 = nn.Conv2d(64, 128, k_size)
        self.conv4 = nn.Conv2d(128, 10, k_size)

        self.conv5 = nn.ConvTranspose2d(10, 128, k_size)
        self.conv6 = nn.ConvTranspose2d(256, 64, k_size)
        self.conv7 = nn.ConvTranspose2d(128, 32, k_size)
        self.conv8 = nn.ConvTranspose2d(64, 3, k_size)

    def forward(self, x):
        out1 = F.relu(self.conv1(x), inplace=True)
        out2 = F.relu(self.conv2(out1), inplace=True)
        out3 = F.relu(self.conv3(out2), inplace=True)
        out4 = F.relu(self.conv4(out3), inplace=True)
        out5 = F.relu(self.conv5(out4), inplace=True)

        out6 = F.relu(self.conv6(torch.cat([out5, out3], 1)), inplace=True)
        out7 = F.relu(self.conv7(torch.cat([out6, out2], 1)), inplace=True)
        out8 = self.conv8(torch.cat([out7, out1], 1))

        return out8


class ReconstructionModel():
    def __init__(self):
        pass

    def initialize(self, opt, filename=None):
        if opt.model == 'normal':
            self.net = ConvAE()
        elif opt.model == 'skip':
            self.net = Skip_ConvAE()
        else:
            raise NotImplementedError('model {} are not implemented!')

        self.opt = opt

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.criterion_l1 = nn.L1Loss()

        if not opt.is_train:
            if opt.auto_test:
                save_filename = filename
            else:
                save_filename = '{}_{}.pth'.format(opt.which_epoch, opt.model)

            self.load_network(self.net, save_filename)


    def forward(self):
        self.out = self.net(self.noised)

    def backward(self):
        self.l1_loss = self.criterion_l1(self.out, self.data)
        self.l1_loss.backward()

    def optimize_parameters(self, data):
        self.data = data
        self.noised = self.add_noise()
        self.data = self.data.to(self.opt.device)
        self.noised = self.data.to(self.opt.device)

        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def add_noise(self):
        noised_data = self.data + torch.randn(self.data.size()) * 0.2

        return noised_data

    # def fgsm_attack(self, epsilon):

    def save_network(self, epoch_label):
        save_filename = '{}.pth'.format(epoch_label)
        save_path = os.path.join(self.opt.checkpoints_path, save_filename)
        torch.save(self.net.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            self.net.to(self.opt.device)

    def load_network(self, network, save_filename):

        save_path = os.path.join(self.opt.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))