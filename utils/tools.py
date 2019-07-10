import torch
import random
import numpy as np

random.seed(2018)

from gpu_flag import device


def cos(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def gvar(x):
    return x.to(device)


