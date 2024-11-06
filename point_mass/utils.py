import torch
import numpy as np


def np_to_torch(x):
    # return torch.from_numpy(x).to(torch.float32).to(device)
    return torch.from_numpy(x).to(torch.float32)


def torch_to_np(x):
    return x.cpu().detach().numpy()