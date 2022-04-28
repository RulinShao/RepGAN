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
    with dnnlib.util.open_url(network_pkl) as f:
        D = legacy.load_network_pkl(f)['D'].to(device)
    return G, D


def reprogramming(
    args
):
    # Load pre-trained models
    G, D = load_pretrained(args.network_pkl)
    G.requires_grad = False
    D.requires_grad = False

    # Initialize mapping modules
    z_map = HiddenMap(G.z_dim).cuda()
    img_map_G = EncDec().cuda()
    img_map_D = EncDec().cuda()

    # Generate images
    z = torch.randn([2, G.z_dim]).cuda()  # latent codes
    z = z_map(z)
    c = 0  # class labels, 0 = no label
    out_G = G(z, c)  # NCHW, float32, dynamic range [-1, +1]
    out_G = img_map_G(out_G)

    # Fake input to D
    fake_img = img_map_D(out_G)
    gen_logits = D(fake_img, c)

    # Real input to D
    real_img = torch.rand_like(fake_img)
    real_logits = D(real_img, c)
    
    # Gmain: Maximize logits for generated images.
    loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

    # Dmain: Minimize logits for generated images.
    loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

    # Dmain: Maximize logits for real images.
    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))

    loss_G = loss_Gmain
    loss_D = loss_Dgen + loss_Dreal

    return


if __name__ == "__main__":
    reprogramming(args)