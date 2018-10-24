import os
from PIL import Image
import random

import torch.utils.data
from torchvision import transforms


def create_dataloader(opt, is_train):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt, is_train)

    return data_loader


class CustomDatasetDataLoader():
    def __init__(self):
        pass

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, is_train):
        self.opt = opt
        self.dataset = CustomDataset()
        self.dataset.initialize(opt, is_train)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=opt.batch_size,
            # shuffle=not opt.serial_batches,
            shuffle=True,
            num_workers=2
        )

    def __iter__(self):
        for i, (data, labels) in enumerate(self.dataloader):
            yield data, labels


class CustomDataset():
    def __init__(self):
        pass

    def name(self):
        return 'CustomDataset'

    def initialize(self, opt, is_train):
        self.opt = opt
        self.root = opt.dataroot
        if opt.phase == 'train':
            if is_train:
                self.data_dir = os.path.join(self.root, 'train')
            else:
                self.data_dir = os.path.join(self.root, 'val')
        elif opt.phase == 'test':
            # self.data_dir = os.path.join(self.root, 'test')
            self.data_dir = os.path.join(self.root, 'val')
        else:
            raise NotImplementedError('phase {} can not be recognized!'.format(opt.phase))

        if not os.path.exists(self.data_dir):
            raise ValueError('{} does not exits'.format(self.data_dir))

        self.n_target = opt.n_target_classes
        self.n_negative = opt.n_negative_classes
        self.n_all = opt.n_all_classes

        if self.n_target + self.n_negative > self.n_all:
            raise ValueError('n_target_classes: {}, n_negative_classess: {} over n_all_classes: {} !'.format(
                self.n_target, self.n_negative, self.n_all
            ))

        classes = torch.arange(0, self.n_all, step=1, dtype=torch.int32)
        # torch.manual_seed(opt.seed)
        # r = torch.randperm(len(classes))
        # classes = classes[r]

        if opt.not_use_face_data:
            tar_classes = classes[:self.n_target]
            if is_train:
                neg_classes = classes[self.n_target: self.n_target+self.n_negative]
            else:
                neg_classes = classes[self.n_target:]
        else:
            tar_classes = classes[:(self.n_target-1)]
            if is_train:
                neg_classes = classes[(self.n_target-1): (self.n_target - 1 + self.n_negative)]
            else:
                neg_classes = classes[self.n_target:]

        tar_paths, tar_labels = self.make_dataset(tar_classes, self.data_dir, is_neg=False)
        tmp_neg_paths, neg_labels = self.make_dataset(neg_classes, self.data_dir, is_neg=True)

        if opt.not_use_face_data:
            pass
        else:
            if opt.phase == 'train':
                if is_train:
                    self.face_data_dir = os.path.join(opt.face_dataroot, 'train')
                else:
                    self.face_data_dir = os.path.join(opt.face_dataroot, 'val')
            elif opt.phase == 'test':
                # self.face_data_dir = os.path.join(opt.face_dataroot, 'test')
                self.face_data_dir = os.path.join(opt.face_dataroot, 'val')

            face_paths, face_labels = self.make_dataset(torch.tensor([0]), self.face_data_dir, is_neg=False, is_face=True)
            face_labels = torch.ones_like(face_labels, dtype=torch.long) * (self.n_target-1)

            tar_labels = torch.cat([tar_labels, face_labels], 0)
            tar_paths += face_paths

        neg_paths = []

        if self.n_negative > 0:
            if opt.use_all_data:
                neg_paths = tmp_neg_paths
            else:
                n_per_class = int(len(tar_labels) / self.n_target)
                n_sampling = n_per_class // len(neg_classes)
                res = n_per_class % len(neg_classes)

                for num in range(len(neg_classes)):
                    begin = num * n_per_class
                    end = num * n_per_class + n_sampling
                    if res > 0:
                        end += 1
                        res -= 1
                    neg_paths += tmp_neg_paths[begin:end]

        random.seed(opt.seed)
        random.shuffle(neg_paths)

        self.paths = tar_paths + neg_paths
        self.labels = torch.cat([tar_labels, neg_labels], 0)

        transform_list = []

        if opt.phase == 'train':
            transform_list.append(transforms.Scale([256, 256], Image.BICUBIC))
            transform_list.append(transforms.RandomCrop(224))
        else:
            transform_list.append(transforms.Scale([224, 224], Image.BICUBIC))

        transform_list += [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # self.transform = transforms.Compose(
        #     [transforms.Scale(256, Image.BICUBIC),
        #      transforms.RandomCrop(224),
        #      transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5),
        #                           (0.5, 0.5, 0.5))]
        # )

    def make_dataset(self, classes, phase_dir, is_neg, is_face=False):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif', 'dng',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

        paths = []
        labels_l = []

        assert os.path.isdir(phase_dir), '{} is not a valid directory'.format(phase_dir)

        cls_dir = sorted(os.listdir(phase_dir))

        for index, cls in enumerate(classes):
            # cls_path = os.path.join(phase_dir, str(cls.item()))
            if is_face:
                cls_path = phase_dir
            else:
                cls_path = os.path.join(phase_dir, str(cls_dir[classes[index].item()]))

            print('{} {}'.format(index, cls_path))

            for root, _, fnames in sorted(os.walk(cls_path)):
                for fname in fnames:
                    if any(fname.endswith(fname) for extension in IMG_EXTENSIONS):
                        path = os.path.join(root, fname)
                        paths.append(path)
                        if is_neg:
                            labels_l.append(self.n_target)
                        else:
                            # relabel to 0 ~ n_target -1 +1 when training
                            labels_l.append(index)

        return paths, torch.tensor(labels_l, dtype=torch.long)

    def __getitem__(self, in_idx):

        idx = in_idx % len(self.paths)

        path = self.paths[idx]
        img_pil = Image.open(path).convert('RGB')
        img = self.transform(img_pil)
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.paths)
