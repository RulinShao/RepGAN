import pickle
from typing import List, Optional
import logging
from tqdm import tqdm
from datetime import datetime
import PIL.Image

import argparse
import torch
import torch.optim as optim
import dnnlib

import legacy

from reprog import *

parser = argparse.ArgumentParser(description='Reprgramming for GAN')
parser.add_argument('--network_pkl', default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl', type=str)
parser.add_argument('--ckpt_path', default='output/afhqcat/2022-04-28/ckpt.pt', type=str)
parser.add_argument('--out_dir', default='out', type=str)

args = parser.parse_args()


def load_model(args):
    G = load_pretrained(args.network_pkl, 'G_ema')
    z_map = HiddenMap(G.z_dim).cuda()
    img_map_G = EncDec(conv_dim=8, repeat_num=1).cuda()
    ckpt = torch.load(args.ckpt_path)
    z_map.load_state_dict(ckpt['z_map'])
    img_map_G.load_state_dict(ckpt['img_map_G'])
    return z_map.cuda(), G.cuda(), img_map_G.cuda()


def load_pretrained(
    network_pkl: str,
    key: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        model = legacy.load_network_pkl(f)[key].to(device)  # type: ignore
    return model


def generate(args, seeds):
    z_map, G, img_map_G = load_model(args)

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        # z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).cuda()
        torch.manual_seed(seed)
        z = torch.randn([1, G.z_dim]).cuda()
        z = z_map(z)
        img = G(z, 0, truncation_psi=1, noise_mode='const')
        img = img_map_G(img)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{args.out_dir}/seed{seed:04d}.png')


if __name__ == '__main__':
    generate(args, list(range(10)))