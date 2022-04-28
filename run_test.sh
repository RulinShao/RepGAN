#!/bin/bash

METFACES=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
AFHQCAT=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl

python test.py --network_pkl $METFACES --batch_size 4