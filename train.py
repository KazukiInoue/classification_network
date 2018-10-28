import argparse
import os
from PIL import Image

import torch
import torch.utils.data
from tensorboardX import SummaryWriter

import data_loader
from data_loader import SplitedDataLoader
from options.train_options import TrainOptions
from tvmodel import TVModel
from custom_dataset_dataloader import create_dataloader

parser = argparse.ArgumentParser(description='Classification')

opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.log_dir))

model = TVModel()
model.initialize(opt)


def train(epoch, total_iter, train_loader):

    model.net.train()
    epoch_iter = 0
    train_loss = 0
    correct = 0

    loader_size = len(train_loader.dataset) * (1 - opt.valid_size)

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

        pred = model.outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        print('epoch {}:, processed {} / {}'.format(epoch, epoch_iter, loader_size))


        if total_iter % opt.save_latest_freq == 0:
            print('saving the latest model (epoch {}, total_iteration {})'.format(epoch, total_iter))
            model.save_network('latest')

    accuracy = correct / loader_size

    print('epoch {} finish.'.format(epoch))
    writer.add_scalars('loss/training_epoch', {'Cross_Entropy': model.ce_loss.item(),
                                               'Accuracy': accuracy}, epoch)

    for param in model.optimizer.param_groups:
        print(param['lr'])

    return total_iter


def validation(epoch, valid_loader):

    loader_size = opt.valid_size * len(valid_loader.dataset)

    with torch.no_grad():
        model.net.eval()

        val_loss = 0
        val_iter = 0
        correct = 0

        for i, (data, labels) in enumerate(valid_loader):
            if data.size(1) == 1:
                data = torch.cat([data, data, data], 1)
            data = data.to(opt.device)
            labels = labels.to(opt.device)
            val_loss += model.validation(data, labels)
            val_iter += 1

            pred = model.outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

        val_loss /= val_iter
        accuracy = correct / loader_size

        writer.add_scalars('loss/validation_epoch', {'Cross_Entropy': val_loss,
                                                     'Accuracy': accuracy}, epoch)
        return val_loss


def main():

    total_iter = 0
    if opt.data_type == 'raw_image':
        train_loader = create_dataloader(opt, is_train=True)
        valid_loader = create_dataloader(opt, is_train=False)
    else:
        train_loader, valid_loader = data_loader.get_train_valid_loader(opt)

    for epoch in range(opt.epochs):
        total_iter = train(epoch, total_iter, train_loader)
        validation(epoch, valid_loader)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch {}, iters{}'.format(epoch, total_iter))
            model.save_network('latest')
            model.save_network(epoch)


if __name__ == '__main__':
    main()