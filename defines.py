import torch
DEVICE = "cuda:0"
# DEVICE="cpu"
IMG_PATH = "/mnt/f/luke_data/data/16bit"
MAX_G_LEVELS = 5
NUM_DOTS = 1000000
LUM_THRESH = 0.25

torch.set_grad_enabled(False)