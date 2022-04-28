import pickle
from typing import List, Optional
import logging

import argparse
from numpy import real
import torch
import torch.optim as optim
import dnnlib

import legacy

from reprog import *

parser = argparse.ArgumentParser(description='Reprgramming for GAN')
parser.add_argument('--network_pkl', help='Pre-trained Network pickle filename', required=True)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--log_interval', default=1)
parser.add_argument('--image_size', default=1024, type=int)

args = parser.parse_args()


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


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
    # Load real images from target dataset
    target_loader = load_pokemon(args.batch_size, args.image_size)

    # Load pre-trained models
    G, D = load_pretrained(args.network_pkl)
    G.requires_grad = False
    D.requires_grad = False

    # Initialize mapping modules
    z_map = HiddenMap(G.z_dim).cuda()
    img_map_G = EncDec().cuda()
    img_map_D = EncDec().cuda()

    # Optimiers for mappings
    optimizer_mapG = optim.SGD(list(z_map.parameters()) + list(img_map_G.parameters()), lr=args.lr, momentum=0.9)
    optimizer_mapD = optim.SGD(img_map_D.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(args.epochs):
        for real_images, _ in target_loader:

            # Generate images
            z = torch.randn([args.batch_size, G.z_dim]).cuda()  # latent codes
            z = z_map(z)
            c = 0  # class labels, 0 = no label
            out_G = G(z, c)  # NCHW, float32, dynamic range [-1, +1]
            out_G = img_map_G(out_G)

            # Fake input to D
            fake_images = img_map_D(out_G)
            gen_logits = D(fake_images, c)

            # Real input to D
            real_images = real_images.cuda()
            real_logits = D(real_images, c)
            
            # Update generator
            optimizer_mapG.zero_grad()
            # Gmain: Maximize logits for generated images.
            loss_G = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
            loss_G.backward()
            optimizer_mapG.step()

            # Update discriminator
            optimizer_mapD.zero_grad()
            # Dmain: Minimize logits for generated images.
            loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            # Dmain: Maximize logits for real images.
            loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
            loss_D = loss_Dgen + loss_Dreal
            loss_D.backward()
            optimizer_mapD.step()
        if epoch % args.log_interval == 0:
            logging.info(f"Epoch:{epoch}, Loss_G:{loss_G:.4f}, Loss_D:{loss_D:.4f}, loss_Dreal:{loss_Dreal:.4f}, loss_Dgen:{loss_Dgen:.4f}")
    
    return z_map, img_map_G, img_map_D


if __name__ == "__main__":
    reprogramming(args)