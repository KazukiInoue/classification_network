import argparse
import os
from PIL import Image

import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

import data_loader
from options.train_options import TrainOptions
from tvmodel import TVModel
from custom_dataset_dataloader import create_dataloader

parser = argparse.ArgumentParser(description='Classification')

opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.runs_dir, opt.name, 'loss'))

model = TVModel()
model.initialize(opt)


def train(epoch, total_iter, train_loader):

    model.net.train()
    epoch_iter = 0
    train_loss = 0

    for i, (data, labels) in enumerate(train_loader):

        if data.size(1) == 1:
            data = torch.cat([data, data, data], 1)
        data = data.to(opt.device)
        labels = labels.to(opt.device)
        batch_size = data.size(0)

        model.optimize_parameters(data, labels)

        total_iter += 1
        epoch_iter += batch_size

        train_loss += model.ce_loss.item()

        # if (i+1) % 10 == 0:
        #     writer.add_scalar('loss/train_CrossEntropy_iteration', train_loss / 10, total_iter)
        #     train_loss = 0

        print('epoch {}:, processed {} / {}'.format(epoch, epoch_iter, len(train_loader.dataset)))

        if total_iter % opt.save_latest_freq == 0:
            print('saving the latest model (epoch {}, total_iteration {})'.format(epoch, total_iter))
            model.save_network('latest')

    print('epoch {} finish.'.format(epoch))
    writer.add_scalar('loss/train_CrossEntropy_epoch', model.ce_loss.item(), epoch)

    return total_iter


def validation(epoch, valid_loader):
    with torch.no_grad():
        # model.net.eval()

        val_loss = 0
        val_iter = 0

        for i, (data, labels) in enumerate(valid_loader):
            if data.size(1) == 1:
                data = torch.cat([data, data, data], 1)
            data = data.to(opt.device)
            labels = labels.to(opt.device)
            val_loss += model.validation(data, labels)
            val_iter += 1

        val_loss /= val_iter
        writer.add_scalar('loss/validation_CrossEntropy_epoch', val_loss, epoch)
        return val_loss


def main():

    total_iter = 0
    if opt.data_type == 'raw_image':
        train_loader = create_dataloader(opt, is_train=True)
        valid_loader = create_dataloader(opt, is_train=False)
    else:
        # transform = transforms.Compose(
        #     [transforms.Scale(224, Image.BICUBIC),
        #      transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5),
        #                           (0.5, 0.5, 0.5))]
        # )
        # kwargs = {'num_workers': 1, 'pin_memory': True} if opt.gpu_id > -1 else {}
        # if opt.data_type == 'CIFAR100':
        train_loader, valid_loader = data_loader.get_train_valid_loader(opt, valid_size=0.2)


            # train_loader = torch.utils.data.DataLoader(
            #     datasets.CIFAR100('../DATASET/CIFAR100', train=True, download=True,
            #                       transform=transform),
            #     batch_size=opt.batch_size, shuffle=True, **kwargs)
            # valid_loader = torch.utils.data.DataLoader(
            #     datasets.CIFAR100('../DATASET/CIFAR100', train=False, transform=transform),
            #     batch_size=opt.batch_size, shuffle=True, **kwargs)
        # elif opt.data_type == 'MNIST':
        #     train_loader = torch.utils.data.DataLoader(
        #         datasets.MNIST('../DATASET/MNIST', train=True, download=True,
        #                        transform=transform),
        #         batch_size=opt.batch_size, shuffle=True, **kwargs)
        #     valid_loader = torch.utils.data.DataLoader(
        #         datasets.MNIST('../DATASET/MNIST', train=False, transform=transform),
        #         batch_size=opt.batch_size, shuffle=True, **kwargs)
        # else:
        #     raise NotImplementedError('{} cannot be recognized!'.format(opt.data_type))

    print('dataset size:', len(train_loader.dataset))

    for epoch in range(opt.epochs):
        total_iter = train(epoch, total_iter, train_loader)
        validation(epoch, valid_loader)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch {}, iters{}'.format(epoch, total_iter))
            model.save_network('latest')
            model.save_network(epoch)


if __name__ == '__main__':
    main()