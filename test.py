import os
import sys

from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.utils.data
from torchvision import datasets, transforms

import data_loader
from options.test_options import TestOptions
from custom_dataset_dataloader import create_dataloader
from classification_model import ClassificationModel
from utill.save_raw_image import save_classified_image

opt = TestOptions().parse()
opt.batch_size = 1



def main(test_loader, model, filename=None):

    correct = 0
    size = 0

    print('now processing')
    for i, (data, labels) in enumerate(test_loader):
        if data.size(1) == 1:
            data = torch.cat([data, data, data], 1)
        data = data.to(opt.device)
        labels = labels.to(opt.device)
        size += labels.size(0)
        model.test(data, labels)

        save_classified_image(i, opt, data, labels.cpu().view(-1).numpy().tolist(), model.fake_list)

        pred = model.outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

    # model.accuracy = torch.tensor(correct / size).to(opt.device)
    accuracy = torch.tensor(correct / size).item()


    # print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    #             correct, size, 100*model.accuracy.item()))

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, size, 100 * accuracy))

    real_labels = model.labels_list[0]
    fake_labels = model.labels_list[1]

    p, r, f, _ = precision_recall_fscore_support(real_labels, fake_labels)

    print('precision: ', p)
    print('recall :', r)
    print('F-measure :', f)

    results_path = os.path.join(opt.results_dir, opt.name)

    if filename:
        epoch = filename.split('_')[0]
    else:
        epoch = opt.which_epoch

    if hasattr(opt, 'use_all_data') and opt.use_all_data:
        save_filename = os.path.join(results_path, epoch+'_test-result_all-data')
    else:
        save_filename = os.path.join(results_path, epoch+'_test-result.txt')

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(save_filename, 'wt') as opt_file:
        opt_file.write('Accuracy:{}, {}/{} (correct / all)\n'.format(
                       accuracy, correct, size))
        opt_file.write('Precision: {} \n'.format(p))
        opt_file.write('Recall: {} \n'.format(r))
        opt_file.write('F-measure: {} \n'.format(f))

        opt_file.write('\n------------ Options -------------\n')
        for k, v in sorted(vars(opt).items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))


if __name__ == '__main__':

    print(opt.name)

    if opt.data_type == 'raw_image':
        test_loader = create_dataloader(opt, is_train=False)
    else:
        test_loader = data_loader.get_test_loader(opt)

    cls_model = ClassificationModel()

    print('data size: ', len(test_loader.dataset))

    if not opt.auto_test:
        model.initialize(opt)
        main(test_loader, model)
    else:
        for file in sorted(os.listdir(opt.save_dir)):
            _, ext = os.path.splitext(file)
            if ext == '.pth':
                print('------loading {}-----'.format(file))
                model.initialize(opt, filename=file)
                main(test_loader, model, file)
