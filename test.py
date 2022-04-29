import pickle
from typing import List, Optional
import logging
from tqdm import tqdm
from datetime import datetime

import argparse
import torch
import torch.optim as optim
import dnnlib

import legacy

from reprog import *

parser = argparse.ArgumentParser(description='Reprgramming for GAN')
parser.add_argument('--network_pkl', help='Pre-trained Network pickle filename', required=True)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--log_interval', default=1)
parser.add_argument('--image_size', default=1024, type=int)
parser.add_argument('--output', default='./output/', type=str)
parser.add_argument('--train_D_interval', default=1, type=int)

args = parser.parse_args()


output_dir = args.output
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

logging.basicConfig(
    filename=os.path.join(output_dir, 'logs.log'),
    filemode='a',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.info(args)


def load_pretrained(
    network_pkl: str,
    key: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        model = legacy.load_network_pkl(f)[key].to(device)  # type: ignore
    return model


def reprogramming(
    args
):
    # Load real images from target dataset
    target_loader = load_pokemon(args.batch_size, args.image_size)

    # Load pre-trained models
    G = load_pretrained(args.network_pkl, 'G_ema').cuda()
    D = load_pretrained(args.network_pkl, 'D').cuda()
    G.requires_grad_(False)
    D.requires_grad_(False)

    # Initialize mapping modules
    z_map = HiddenMap(G.z_dim).cuda()
    img_map_G = EncDec(conv_dim=8, repeat_num=1).cuda()
    img_map_D = EncDec(conv_dim=8, repeat_num=1).cuda()

    # Optimiers for mappings
    # optimizer_mapG = optim.SGD(list(z_map.parameters()) + list(img_map_G.parameters()), lr=args.lr, momentum=0.9)
    # optimizer_mapD = optim.SGD(img_map_D.parameters(), lr=args.lr, momentum=0.9)
    optimizer_mapG = optim.Adam(list(z_map.parameters()) + list(img_map_G.parameters()), lr=args.lr, betas=[0, 0.99])
    optimizer_mapD = optim.Adam(img_map_D.parameters(), lr=args.lr, betas=[0, 0.99])

    for epoch in range(args.epochs):
        for i, (real_images, _) in enumerate(target_loader):

            # Generate images
            z = torch.randn([args.batch_size, G.z_dim]).cuda()  # latent codes
            z = z_map(z)
            c = 0  # class labels, 0 = no label
            out_G = G(z, c)  # NCHW, float32, dynamic range [-1, +1]
            out_G = img_map_G(out_G)

            # Fake input to D (with gradient to G)
            fake_images = img_map_D(out_G)
            gen_logits = D(fake_images, c)
            
            optimizer_mapG.zero_grad()
            
            # Gmain: Maximize logits for generated images.
            loss_G = torch.nn.functional.softplus(-gen_logits).mean() # -log(sigmoid(gen_logits))
            loss_G.backward(retain_graph=True)

            optimizer_mapG.step()

            if epoch % args.train_D_interval == 0:
                # Fake input to D (without gradient to G)
                fake_images_ = img_map_D(out_G.detach())
                gen_logits_ = D(fake_images_, c)

                # Real input to D
                real_images = real_images.cuda()
                real_images = img_map_D(real_images)
                real_logits = D(real_images, c)

                optimizer_mapD.zero_grad()

                # Dmain: Minimize logits for generated images.
                loss_Dgen = torch.nn.functional.softplus(gen_logits_).mean() # -log(1 - sigmoid(gen_logits))
                # Dmain: Maximize logits for real images.
                loss_Dreal = torch.nn.functional.softplus(-real_logits).mean() # -log(sigmoid(real_logits))
                loss_D = loss_Dgen + loss_Dreal
                loss_D.backward()

                optimizer_mapD.step()

        
        if epoch % args.log_interval == 0:
            if epoch % args.train_D_interval == 0:
                logging.info(f"Epoch:{epoch}, Loss_G:{loss_G:.4f}, Loss_D:{loss_D:.4f}, loss_Dreal:{loss_Dreal:.4f}, loss_Dgen:{loss_Dgen:.4f}")
            else:
                logging.info(f"Epoch:{epoch}, Loss_G:{loss_G:.4f}")
            save_ckpt(args, z_map, img_map_G, img_map_D, optimizer_mapD, optimizer_mapG)
    
    return z_map, img_map_G, img_map_D, optimizer_mapD, optimizer_mapG


def save_ckpt(args, z_map, img_map_G, img_map_D, optimizer_mapD, optimizer_mapG):
    PATH = os.path.join(output_dir, 'ckpt.pt')
    torch.save({
            'z_map': z_map.state_dict(),
            'img_map_G': img_map_G.state_dict(),
            'img_map_D': img_map_D.state_dict(),
            'optimizer_mapD': optimizer_mapD.state_dict(),
            'optimizer_mapG': optimizer_mapG.state_dict(),
            }, PATH)

if __name__ == "__main__":
    z_map, img_map_G, img_map_D, opt_D, opt_G = reprogramming(args)
    
