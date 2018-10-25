# please refer to here: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

import torch
import numpy as np
from PIL import Image

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class SplitedDataLoader():
    def __init__(self):
        pass

    def name(self):
        return 'SplitedDataLoader'

    def initialize(self, opt, is_train, valid_size=0.2):
        self.opt = opt
        self.is_train = is_train
        self.valid_size = valid_size
        self.dataloader = get_train_valid_loader(opt, is_train)

    def __iter__(self):
        for i, (data, labels) in enumerate(self.dataloader):
            yield data, labels

    def __len__(self):
        split = int(np.floor(self.valid_size * len(self.dataloader.dataset)))
        print(len(self.dataloader.dataset))
        if self.is_train:
            return len(self.dataloader.dataset) - split
        else:
            return split


def get_train_valid_loader(opt,
                           random_seed=0,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((opt.valid_size >= 0) and (opt.valid_size <= 1)), error_msg

    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    normalize = transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    # define transforms
    if opt.model == 'lenet':
        resize = 32
    else:
        resize = 224

    train_transform = transforms.Compose([
            transforms.Scale(resize, Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = train_transform

    # load the dataset
    if opt.data_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=opt.dataroot, train=True,
            download=True, transform=train_transform,
        )
        valid_dataset = datasets.CIFAR10(
            root=opt.dataroot, train=True,
            download=True, transform=valid_transform,
        )
    elif opt.data_type == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            root=opt.dataroot, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = datasets.CIFAR100(
            root=opt.dataroot, train=True,
            download=True, transform=valid_transform,
        )
    else:
        raise NotImplementedError('{} are not implemented. We can use CIFAR10 and CIFAR100'.format(opt.data_type))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(opt.valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    # if show_sample:
    #     sample_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=9, shuffle=shuffle,
    #         num_workers=num_workers, pin_memory=pin_memory,
    #     )
    #     data_iter = iter(sample_loader)
    #     images, labels = data_iter.next()
    #     X = images.numpy().transpose([0, 2, 3, 1])
    #     plot_images(X, labels)

    # if is_train:
    #     return train_loader
    # else:
    #     return valid_loader

    return train_loader, valid_loader


def get_test_loader(opt,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )

    normalize = transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    # define transform
    transform = transforms.Compose([
        transforms.Scale(224, Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.data_type == 'CIFAR10':
        dataset = datasets.CIFAR10(
            root=opt.dataroot, train=False,
            download=True, transform=transform,
        )
    elif opt.data_type == 'CIFAR100':
        dataset = datasets.CIFAR100(
            root=opt.dataroot, train=False,
            download=True, transform=transform,
        )
    else:
        raise NotImplementedError('{} are not implemented. We can use CIFAR10 and CIFAR100'.format(data_type))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader