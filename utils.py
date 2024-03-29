import torch
from torch.nn import functional as F
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from typing import *
from defines import *


def gkern(l, sig):
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def to_rgb(img: torch.Tensor):
    img = img.permute(1, 2, 0) / (img.max())
    return img.to("cpu")


def diff_of_g(img: torch.Tensor, g):
    gpu_img = img.to(DEVICE)
    img = F.conv2d(gpu_img, g.unsqueeze(1), padding="same", groups=3).squeeze()
    img = img.to("cpu")
    del gpu_img
    return img


def get_kerns(size=9):
    assert size % 2 != 0

    x = (torch.linspace(0, 1, steps=size, device=DEVICE).round() * 2 - 1).expand(size=(size, size))
    x[:, int(size / 2)] = 0
    x = x
    y = x.transpose(dim0=1, dim1=0)
    energy = torch.ones_like(x) / (size**2)
    basline = torch.asarray(gkern(size * 2 + 1, 2.0), dtype=torch.float32, device=DEVICE)
    basline -= basline.mean()
    return x, y, energy, basline


def load_image(path: str):
    # im = Image.open()
    np_array = tifffile.imread(path)
    img = torch.asarray(np_array.astype(np.float32), dtype=torch.float32).permute(
        2, 0, 1
    )
    img = F.interpolate(
        img.unsqueeze(0), scale_factor=(1.0, 1.0), mode="bilinear", align_corners=False
    ).squeeze()
    return img
