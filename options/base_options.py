import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--dataroot', type=str, required=True,
                            help='data to use e.g. ./dataset/raw_image/CIFAR100')
        self.parser.add_argument('--data_type', type=str, default='raw_image',
                            help='data set to use [raw_image | CIFAR100 | MNIST]')
        self.parser.add_argument('--face_dataroot', type=str, default='../DATASET/img_align_celeba_2000/')
        self.parser.add_argument('--not_use_face_data', action='store_true',
                                 help='do not use face data, only use dataroot dataset')

        self.parser.add_argument('--model', type=str, default='resnet50',
                            help='dimension of latent space [vgg11,13,16,19,13bn.. | resnet18,34,50,101,152 ]')

        self.parser.add_argument('--n_all_classes', type=int, required=True,
                            help='number of classes excluding face data')
        self.parser.add_argument('--n_target_classes', type=int, required=True,
                            help='number of target classes excluding face data')
        self.parser.add_argument('--n_negative_classes', type=int, default=0,
                            help='number of classes as negatives')
        self.parser.add_argument('--use_all_data', action='store_true',
                                 help='use all images for each class')

        self.parser.add_argument('--ranodm_seed', type=int, default=0,
                            help='seed for split target or negative classes')

        self.parser.add_argument('--batch_size', type=int, default=64,
                            help='input batch size for training (default: 128)')
        self.parser.add_argument('--seed', type=int, default=1,
                            help='random seed (default: 1)')
        self.parser.add_argument('--log_interval', type=int, default=10,
                            help='how many batches to wait before logging training status')
        self.parser.add_argument('--augumentation', action='store_true',
                                 help='use data augumentation')

        self.parser.add_argument('--emb_interval', type=int, default=5,
                            help='how many epoch to wait before visualizing embedding')

        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id: e.g. 0. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--name', required=True, type=str, help='name of experiment')

        self.initialized = True

    def parse(self):

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if self.is_train:
            if os.path.exists(self.opt.save_dir):
                raise ValueError('[%s] aleready exists!' % self.opt.save_dir)
            else:
                os.makedirs(self.opt.save_dir)

        self.opt.is_train = self.is_train
        self.opt.device = torch.device('cuda:' + str(self.opt.gpu_id)
                                       if torch.cuda.is_available() and self.opt.gpu_id > -1
                                       else 'cpu')

        args = vars(self.opt)
        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        file_path = os.path.join(self.opt.save_dir, 'train_opt.txt')
        with open(file_path, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.opt