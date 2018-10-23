import argparse
import numpy as np
from PIL import Image
import os

import torch
import torch.utils.data
from torchvision import datasets, transforms


def tensor2im(image_tensor, imtype=np.uint8):
    image_np = image_tensor[0].cpu().float().numpy()
    if image_np.shape[0] == 1:
        image_np = np.tile(image_np, (3, 1, 1))
    image_np = (np.transpose(image_np, (1, 2, 0)) + 1) / 2 * 255.0

    return image_np.astype(imtype)


def save_image(args, dataloader, phase):

    save_root_dir = '../dataset/raw_image/'+args.data_type+phase
    save_root_dir = os.path.join('../dataset/raw_image', args.data_type, phase)

    for i, (data, label) in enumerate(dataloader):

        img_np = tensor2im(data)

        label_s = label[0].item()
        save_dir = os.path.join(save_root_dir, str(label_s))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_name = '{}_{}.png'.format(i, label_s)
        img_path = os.path.join(save_dir, img_name)

        img_pil = Image.fromarray(img_np)
        img_pil.save(img_path)

        if i % 500 == 0:
            print('{} data: {} / {} are saved.'.format(phase, i, len(dataloader.dataset)))

    print('All {} data are saved'.format(phase))


def save_classified_image(iteration, opt, data, real_labels, fake_labels):

    save_root_dir = os.path.join(opt.results_dir, opt.name, opt.which_epoch)

    for real_label, fake_label in zip(real_labels, fake_labels):
        img_np = tensor2im(data)

        if real_label == fake_label:
            save_dir = os.path.join(save_root_dir, 'correctly_classified', str(real_label))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_name = '{}.png'.format(iteration)
        else:
            save_dir = os.path.join(save_root_dir, 'miss_classified', str(fake_label))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_name = '{}_missclassify_{}_as_{}.png'.format(iteration, real_label, fake_label)

        img_path = os.path.join(save_dir, img_name)
        img_pil = Image.fromarray(img_np)
        img_pil.save(img_path)


def main():

    parser = argparse.ArgumentParser('Download image from torchvision.datasets')
    parser.add_argument('--data_type', type=str, default='CIFAR100',
                        help='which dataset to download')
    args = parser.parse_args()

    transform = transforms.Compose(
         [transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5),
                               (0.5, 0.5, 0.5))]
    )

    if args.data_type == 'CIFAR100':
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../dataset/CIFAR100', train=True, download=True, transform=transform),
                shuffle=False, batch_size=1, num_workers=2)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../dataset/CIFAR100', train=False, transform=transform),
            shuffle=False, batch_size=1, num_workers=2)
    elif args.data_type == 'CIFAR10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../dataset/CIFAR10', train=True, download=True, transform=transform),
            shuffle=False, batch_size=1, num_workers=2)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../dataset/CIFAR10', train=False, transform=transform),
            shuffle=False, batch_size=1, num_workers=2)
    else:
        raise NotImplementedError('{} cannot be downloaded !'.format(args.data_type))

    save_image(args, train_loader, phase='train')
    save_image(args, test_loader, phase='test')


if __name__ == '__main__':

    main()
    print('all images are saved.')