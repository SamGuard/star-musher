import torch
from torch.nn import functional as F
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from typing import *
from functorch import vmap

from utils import *
from defines import *

# Define guassian levels
G_SIZE = 31
G_LIST: List[torch.Tensor] = [
    torch.asarray(gkern(G_SIZE, 1.0 * (2**i)), dtype=torch.float32, device=DEVICE)
    .unsqueeze(2)
    .expand(-1, -1, 3)
    .permute(2, 0, 1)
    for i in range(6)
]


def too_close(dots: torch.Tensor):
    n_dots = dots.shape[1]
    for i in range(n_dots):
        d = dots[:, i]
        if d[0].item() == -1 or d[1].item() == -1:
            continue
        dist = ((dots - d.unsqueeze(1).expand((-1, n_dots))) ** 2).sum(dim=0).sqrt()
        is_too_close = dist < 10.0
        is_too_close[i] = False
        dots[:, is_too_close.nonzero()] = -1.0
    dots = dots[:, (dots[0] > 0.0).nonzero().squeeze()]
    return dots


def feature_extract(dots: torch.Tensor, d_index: torch.Tensor):
    n_dots = dots.shape[1]
    n_attr = 5
    d: torch.Tensor = dots[:, d_index].squeeze()
    dist = ((dots - d.unsqueeze(1).expand((-1, n_dots))) ** 2).sum(dim=0).sqrt()

    close_stars: torch.Tensor = dots[:, dist.argsort()][:, 1 : n_attr + 1]
    print(close_stars)


# Detect stars
def star_detect(rgb_img: torch.Tensor):
    # Convert to BW image and squash
    img = rgb_img.to(device=DEVICE)
    img = img.sum(dim=0) / (3 * img.max())
    # Clamp values and then stretch
    img = (img.clamp(min=LUM_THRESH) - LUM_THRESH) / (1 - LUM_THRESH)
    dims = img.shape
    print(dims)

    dot_map: torch.Tensor = torch.zeros_like(img, dtype=torch.uint8)
    dots = torch.stack(
        (
            torch.randint(0, dims[0] - 1, (NUM_DOTS,), dtype=torch.long),
            torch.randint(0, dims[1] - 1, (NUM_DOTS,), dtype=torch.long),
        ),
        dim=0,
    ).to(device=DEVICE)
    kern_grad_y, kern_grad_x, kern_energy, kern_baseln = get_kerns(size=5)
    img = F.conv2d(
        img.unsqueeze(0).unsqueeze(0),
        kern_baseln.unsqueeze(0).unsqueeze(0),
        padding="same",
        groups=1,
    ).squeeze()
    img += img.min()
    img /= img.max()
    img_grad_x = F.conv2d(
        img.unsqueeze(0).unsqueeze(0),
        kern_grad_x.unsqueeze(0).unsqueeze(0),
        padding="same",
        groups=1,
    ).squeeze()
    img_grad_y = F.conv2d(
        img.unsqueeze(0).unsqueeze(0),
        kern_grad_y.unsqueeze(0).unsqueeze(0),
        padding="same",
        groups=1,
    ).squeeze()
    img_energy = F.conv2d(
        img.unsqueeze(0).unsqueeze(0),
        kern_energy.unsqueeze(0).unsqueeze(0),
        padding="same",
        groups=1,
    ).squeeze()

    """
    blur = (
        torch.asarray(gkern(G_SIZE, 10), dtype=torch.float32, device=DEVICE)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    img_grad_x = F.conv2d(
        img_grad_x.unsqueeze(0).unsqueeze(0),
        blur,
        padding="same",
        groups=1,
    ).squeeze()
    img_grad_y = F.conv2d(
        img_grad_y.unsqueeze(0).unsqueeze(0),
        blur,
        padding="same",
        groups=1,
    ).squeeze()
    """
    scale_x = img_grad_x.abs().max()
    scale_y = img_grad_y.abs().max()
    scale_e = img_energy.abs().max()
    img_grad_x /= scale_x
    img_grad_y /= scale_y
    img_energy /= scale_e
    for i in range(100):
        diff_x = (img_grad_x[dots[0], dots[1]] * 1000).round().to(dtype=torch.long)
        diff_y = (img_grad_y[dots[0], dots[1]] * 1000).round().to(dtype=torch.long)
        diff_x = diff_x.clamp(-1.0, 1.0)
        diff_y = diff_y.clamp(-1.0, 1.0)
        print(diff_x.abs().sum(), diff_y.abs().sum())
        dots[0] = (dots[0] + diff_x).clamp(0, dims[0] - 1)
        dots[1] = (dots[1] + diff_y).clamp(0, dims[1] - 1)

    dot_energy: torch.Tensor = img_energy[dots[0], dots[1]]
    e_threshhold = torch.quantile(dot_energy, 0.99)

    is_high_energy = dot_energy.ge(e_threshhold)
    high_energy_dots = torch.stack(
        (dots[0].masked_select(is_high_energy), dots[1].masked_select(is_high_energy)),
        dim=0,
    )
    print(is_high_energy.sum())
    print(high_energy_dots.shape)

    high_energy_dots = too_close(high_energy_dots)
    print(f"Filtered shape={high_energy_dots.shape}")
    feature_extract(high_energy_dots, 0)

    plt.figure(0)
    plt.imshow(img_grad_x.cpu())
    plt.figure(1)
    plt.imshow(img_grad_y.cpu())
    plt.figure(2)
    plt.imshow(img_energy.cpu())
    plt.figure(3)
    plt.imshow(to_rgb(rgb_img))
    plt.scatter(high_energy_dots[1].cpu(), high_energy_dots[0].cpu(), s=3, c="red")
    plt.show()
    # plt.pause(1)


def main():
    img = load_image(f"{IMG_PATH}/ELEPHANT TRUNK NEBULA-FL36MM-F3.2-SHO.tiff")
    print(img.shape)
    star_detect(diff_of_g(img, G_LIST[1]))


main()


"""
Code store
for i in range(len(G_LIST) - 1):
        diff = diff_of_g(img, G_LIST[i]) - diff_of_g(img, G_LIST[i + 1])
        diff = (diff - diff.min()) / 2
        print(diff)
        blur = to_rgb(diff)
        print(blur.shape)
        plt.figure(i)
        plt.imshow(blur)
    plt.show()

"""
