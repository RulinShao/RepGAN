#!/bin/bash

METFACES=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
AFHQCAT=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl
CIFAR10=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

# python test.py --network_pkl $METFACES --batch_size 2 --output output/metfaces/ --image_size 1024
python test.py --network_pkl $AFHQCAT --batch_size 4 --output output/afhqcat2/ --image_size 512 --train_D_interval 5
# python test.py --network_pkl $CIFAR10 --batch_size 32 --output output/cifar10/ --image_size 32