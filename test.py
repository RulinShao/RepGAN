import pickle
from typing import List, Optional

import argparse
import torch
import dnnlib

import legacy

from reprog import *

parser = argparse.ArgumentParser(description='Reprgramming for GAN')
parser.add_argument('--network_pkl', help='Pre-trained Network pickle filename', required=True)

args = parser.parse_args()


def load_pretrained(
    network_pkl: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    return G


def reprogramming(
    args
):
    # Generate image
    G = load_pretrained(args.network_pkl)
    z = torch.randn([2, G.z_dim]).cuda()  # latent codes
    c = 0  # class labels (not used in this example)
    img = G(z, c)  # NCHW, float32, dynamic range [-1, +1]
    map = EncDec().cuda()
    out = map(img)
    return






if __name__ == "__main__":
    reprogramming(args)