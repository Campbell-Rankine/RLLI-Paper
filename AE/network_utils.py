from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F

def get_loss_fn(lossarg: str) -> callable:
    if lossarg == 'BCE':
        return T.nn.BCEWithLogitsLoss()
    elif lossarg == 'MSE':
        return T.nn.MSELoss()
    else:
        print("%s is an invalid loss fn" % lossarg)
        raise AttributeError

def target(window, ticker_data):
    """
    Target is risk/reward balanced return function, can be made more complicated if necessary

    This is the target from representation based clustering RL-Algo
    """
    section = ticker_data['close']
    stdev = np.std(section[-window:-1])
    return (section[-1] - section[-window]) / (section[-window] * stdev)

def compute_target_column(ticker_data):
    """
    precompute return values across all available training timesteps
    """
    raise NotImplementedError

class Regularizer(nn.Module):
    def __init__(self, lamb):
        super(Regularizer, self).__init__()
        self.lamb = lamb

    def forward(self, grad):
        return self.lamb * (T.sum(T.norm(grad, 'fro'), dim=0))

loss_fn = nn.MSELoss()

from torch.nn.parameter import Parameter
class EqualizedLR_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = np.sqrt(2/(in_ch * kernel_size[0] * kernel_size[1]))

        self.weight = Parameter(T.Tensor(out_ch, in_ch, *kernel_size))
        self.bias = Parameter(T.Tensor(out_ch))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        return F.conv2d(x, self.weight*self.scale, self.bias, self.stride, self.padding).squeeze(-1)

from torchvision import transforms
class input_transform(nn.Module):
    """
    Learned input transform for the encoder model
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.cvt = EqualizedLR_Conv2d(in_c, out_c, (1,1), stride=(1,1))
        
    def forward(self, x):
        return F.interpolate(self.cvt(x), (64,64))

class output_transform(nn.Module):
    """
    Learned input transform for the encoder model
    """
    def __init__(self, in_c, out_c, dim):
        super().__init__()
        args = {'in_channels': in_c, 'out_channels': out_c, 
                    'kernel_size' : (1,1), 'stride' : (1,1)}
        self.cvt_r = nn.ConvTranspose2d(**args)
        self.dim = dim
        
    def forward(self, x):
        return F.interpolate(self.cvt_r(x), self.dim)