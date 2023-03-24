"""
MODELS TAKEN FROM GITHUB REPO: https://github.com/MishaLaskin/vqvae

Author: Misha Laskin
Model Name: VQVAE
Libraries: PyTorch (and other standard ML imports)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_args():
    import json

    parser = argparse.ArgumentParser()

    ### - Global Params - ###
    parser.add_argument("-debug", "--debug", dest="debug", metavar="debug", default = False,
                        type=bool, help="debug flag, minimize data to make things quicker to debug")
    parser.add_argument("-e", "--e", dest="e", metavar="epochs", default = 64,
                        type=int, help="default num epochs")

    ### - AE Args - ###
    parser.add_argument("-latent", "--latent", dest="latent", metavar="latent", default = 138,
                        type=int, help="latent size")
    parser.add_argument("-window", "--window", dest="window", metavar="window", default = 30,
                        type=int, help="default window training size")
    
    ### - Misc. Args - ###
    parser.add_argument("-verbose", "--verbose", dest="verbose", metavar="verbose", default = False,
                        type=bool, help="Print Env info")
    
    args = parser.parse_args()

    return args

from torch.nn.parameter import Parameter
import torch as T
class EqualizedLR_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_c = in_ch
        self.out_c = out_ch
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

class input_transform(nn.Module):
    """
    Learned input transform for the encoder model
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.cvt = EqualizedLR_Conv2d(in_c, out_c, (1,1), stride=(1,1))
        
    def forward(self, x):
        return self.cvt(x)
    
class output_transform(nn.Module):
    """
    Learned input transform for the encoder model
    """
    def __init__(self, in_transform: input_transform):
        super().__init__()
        args = {'in_channels': in_transform.cvt.in_c, 'out_channels': in_transform.cvt.out_c, 
                    'kernel_size' : (1,1), 'stride' : (1,1)}
        self.cvt_r = nn.ConvTranspose2d(**args)

    def forward(self, x):
        return self.cvt_r(x)

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x
    
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()
    # test Residual Layer
    res = ResidualLayer(40, 40, 20)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    # test res stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)