import argparse
import os
from PIL import Image

import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from options.train_options import TrainOptions
from tvmodel import TVModel
from custom_dataset_dataloader import create_dataloader

parser = argparse.ArgumentParser(description='Classification')

opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.runs_dir, opt.name, 'loss'))

model = TVModel()
model.initialize(opt)


def train(epoch, total_iter, trainloader):

    model.net.train()
    epoch_iter = 0
    train_loss = 0

    for i, (data, labels) in enumerate(trainloader):

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

        print('epoch {}:, processed {} / {}'.format(epoch, epoch_iter, len(trainloader.dataset)))

        if total_iter % opt.save_latest_freq == 0:
            print('saving the latest model (epoch {}, total_iteration {})'.format(epoch, total_iter))
            model.save_network('latest')

    print('epoch {} finish.'.format(epoch))
    writer.add_scalar('loss/train_CrossEntropy_epoch', model.ce_loss.item(), epoch)

    return total_iter


def validation(epoch, validloader):
    with torch.no_grad():
        # model.net.eval()

        val_loss = 0
        val_iter = 0

        for i, (data, labels) in enumerate(validloader):
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

    print('hoge')

    total_iter = 0
    if opt.data_type == 'raw_image':
        trainloader = create_dataloader(opt, is_train=True)
        validloader = create_dataloader(opt, is_train=False)
    else:
        transform = transforms.Compose(
            [transforms.Scale(224, Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))]
        )
        kwargs = {'num_workers': 1, 'pin_memory': True} if opt.gpu_id > -1 else {}
        if opt.data_type == 'CIFAR100':
            trainloader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../DATASET/CIFAR100', train=True, download=True,
                                  transform=transform),
                batch_size=opt.batch_size, shuffle=True, **kwargs)
            validloader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../DATASET/CIFAR100', train=False, transform=transform),
                batch_size=opt.batch_size, shuffle=True, **kwargs)
        elif opt.data_type == 'MNIST':
            trainloader = torch.utils.data.DataLoader(
                datasets.MNIST('../DATASET/MNIST', train=True, download=True,
                               transform=transform),
                batch_size=opt.batch_size, shuffle=True, **kwargs)
            validloader = torch.utils.data.DataLoader(
                datasets.MNIST('../DATASET/MNIST', train=False, transform=transform),
                batch_size=opt.batch_size, shuffle=True, **kwargs)
        else:
            raise NotImplementedError('{} cannot be recognized!'.format(opt.data_type))

    print('dataset size:', len(trainloader.dataset))

    for epoch in range(opt.epochs):
        total_iter = train(epoch, total_iter, trainloader)
        validation(epoch, validloader)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch {}, iters{}'.format(epoch, total_iter))
            model.save_network('latest')
            model.save_network(epoch)


if __name__ == '__main__':
    main()