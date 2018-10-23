## train
 python train.py --dataroot ./dataset/raw_image/CIFAR100 --data_type CIFAR100_image --batch_size 64 --epochs 50 --model alexnet --n_target_classes 90 --n_negative_classes 10 --name alexnet_50epoch_90-10

 ## test
 python test.py --dataroot ./dataset/raw_image/CIFAR100 --data_type CIFAR100_image --model alexnet --n_target_classes 90 --n_negative_classes 10 --which_epoch latest --name alexnet_50epoch_90-10