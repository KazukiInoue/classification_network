## train
 python train.py --dataroot ./dataset/raw_image/CIFAR100 --data_type CIFAR100_image --batch_size 64 --epochs 50 --model alexnet --n_target_classes 90 --n_negative_classes 10 --name alexnet_50epoch_90-10
 
 10/25~
  python train.py --dataroot ../DATASET/CIFAR10 --data_type CIFAR10 --batch_size 64 --epochs 100 --model lenet --n_all 10 --n_target_classes 10 --name cifar10_lenet_bs-64

 ## test
 python test.py --dataroot ./dataset/raw_image/CIFAR100 --data_type CIFAR100_image --model alexnet --n_target_classes 90 --n_negative_classes 10 --which_epoch latest --name alexnet_50epoch_90-10
 
 10/25 ~
python test.py --dataroot ../DATASET/CIFAR10 --data_type CIFAR10 --model alexnet --n_all_classes 10 --n_target_classes 10   --which_epoch latest --name cifar10_alexnet_bs-64