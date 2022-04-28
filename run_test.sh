#!/bin/bash

METFACES=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
AFHQCAT=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl

# python test.py --network_pkl $METFACES --batch_size 4 --output output/metfaces/ --image_size 1024
python test.py --network_pkl $AFHQCAT --batch_size 4 --output output/afhqcat/ --image_size 512