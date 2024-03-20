import numpy as np
import imageio.v2 as imageio
import torch
import random

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mse_fn(pred, gt):
    return ((pred - gt) ** 2).mean()

def psnr_fn(pred, gt):
    return -10. * torch.log10(mse_fn(pred, gt))


def read_image(im_path):
    im = imageio.imread(im_path)
    im = np.array(im).astype(np.float32) / 255.
    return im

def write_image(im_path, im):
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(im_path, im)

