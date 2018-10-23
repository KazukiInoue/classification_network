import os
import numpy as np

import torch
from torch import nn
from torchvision import models


class TVModel():

    def initialize(self, opt, filename=None):
        self.opt = opt
        self.gpu_id = opt.gpu_id
        # self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if opt.data_type == 'raw_image':
            if opt.n_negative_classes == 0:
                self.n_classes = opt.n_target_classes
            else:
                self.n_classes = opt.n_target_classes + 1
        elif opt.data_type == 'MNIST' or opt.data_type == 'CIFAR10':
            self.n_classes = 10
        elif opt.data_type == 'CIFAR100':
            self.n_classes =100

        self.net = self.set_networks()
        self.net.to(opt.device)
        self.criterion_ce = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.net.parameters(),
        #                                   lr=0.002, betas=(0.5, 0.999))
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        if not opt.is_train:
            if opt.auto_test:
                save_filename = filename
            else:
                save_filename = '{}_{}.pth'.format(opt.which_epoch, opt.model)

            self.load_network(self.net, save_filename)

        if not opt.is_train:
            self.labels_list = [[], []]

        # print('---------- Networks initialized -------------')
        # self.print_network()
        # print('-----------------------------------------------')

    def set_networks(self):

        if self.opt.model == 'alexnet':
            net = models.AlexNet(num_classes=self.n_classes)

        elif self.opt.model == 'vgg11':
            net = models.vgg11(num_classes=self.n_classes)
        elif self.opt.model == 'vgg11_bn':
            net = models.vgg11_bn(num_classes=self.n_classes)
        elif self.opt.model == 'vgg13':
            net = models.vgg13(num_classes=self.n_classes)
        elif self.opt.model == 'vgg13_bn':
            net = models.vgg13_bn(num_classes=self.n_classes)
        elif self.opt.model == 'vgg16':
            net = models.vgg16(num_classes=self.n_classes)
        elif self.opt.model == 'vgg16_bn':
            net = models.vgg16_bn(num_classes=self.n_classes)
        elif self.opt.model == 'vgg19':
            net = models.vgg19(num_classes=self.n_classes)
        elif self.opt.model == 'vgg19_bn':
            net = models.vgg19_bn(num_classes=self.n_classes)

        elif self.opt.model == 'resnet18':
            net = models.resnet18(num_classes=self.n_classes)
        elif self.opt.model == 'resnet34':
            net = models.resnet34(num_classes=self.n_classes)
        elif self.opt.model == 'resnet50':
            net = models.resnet50(num_classes=self.n_classes)
        elif self.opt.model == 'resnet101':
            net = models.resnet101(num_classes=self.n_classes)
        elif self.opt.model == 'resnet152':
            net = models.resnet152(num_classes=self.n_classes)
        else:
            raise NotImplementedError('model layer_num={} is not recognized!'
                                      .format(self.opt.model))

        return net

    def forward(self, data):
        self.outputs = self.net(data)

    def backward(self, labels):
        self.ce_loss = self.criterion_ce(self.outputs, labels)
        self.ce_loss.backward()

    def optimize_parameters(self, data, labels):
        self.optimizer.zero_grad()
        self.forward(data)
        self.backward(labels)
        self.optimizer.step()

    def validation(self, data, labels):
        self.forward(data)
        return self.criterion_ce(self.outputs, labels).item()

    def test(self, data, labels):

        self.net.eval()
        self.forward(data)

        real_labels = labels.data
        fake_labels = self.outputs.data

        real_list = real_labels.cpu().view(-1).numpy().tolist()

        fake_np = fake_labels.cpu().numpy()
        self.fake_list = np.argmax(fake_np, axis=1).tolist()

        self.labels_list[0] += real_list
        self.labels_list[1] += self.fake_list

    def save_network(self, epoch_label):
        save_filename = '{}_{}.pth'.format(epoch_label, self.opt.model)
        save_path = os.path.join(self.opt.save_dir, save_filename)
        torch.save(self.net.cpu().state_dict(), save_path)
        if self.opt.gpu_id > -1 and torch.cuda.is_available():
            self.net.to(self.opt.device)

    # def load_network(self, network, network_label, epoch_label):
    def load_network(self, network, save_filename):

        # if filename:
        #     save_filename = '{}_{}.pth'.format(epoch_label, network_label)
        # else:
        #     save_filename = filename
        #
        save_path = os.path.join(self.opt.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def print_network(self):
        num_params = 0
        for param in self.net.parameters():
            num_params += param.numel()
        print(self.net)
        print('Total number of parameters: %d' % num_params)