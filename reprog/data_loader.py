
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn, optim
from torchvision import datasets
from matplotlib import pyplot as plt
import seaborn as sns


device = 'cuda' if torch.cuda.is_available() else 'cpu'
POKEMON_NORM = norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def load_pokemon(batch_size, img_size=256):
    path = '/home/hh239/rulin/dataset/pokemon-images-dataset/pokemon'
    assert os.path.exists(path)
    norm = POKEMON_NORM
    transf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm,inplace=True),
        ])
    dataset = datasets.ImageFolder(root=path,transform=transf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader


# Visualization
def unnorm(images, means, stds):
    means = torch.tensor(means).reshape(1,3,1,1)
    stds = torch.tensor(stds).reshape(1,3,1,1)
    return images*stds+means

def show_batch(images, norm, batch_size):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xticks([]); ax.set_yticks([])
    unnorm_images = unnorm(images, *norm)
    ax.imshow(make_grid(unnorm_images[:batch_size], nrow=8).permute(1, 2, 0).clamp(0,1))
    fig.savefig('pokemon.png')


if __name__ == '__main__':
    data_loader = load_pokemon(64)
    print(len(data_loader))
    for img, _ in data_loader:
        show_batch(img, POKEMON_NORM, 64)
        break