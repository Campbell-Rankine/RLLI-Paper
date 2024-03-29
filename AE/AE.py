"""
MODELS TAKEN FROM GITHUB REPO: https://github.com/MishaLaskin/vqvae

Author: Misha Laskin
Model Name: VQVAE
Libraries: PyTorch (and other standard ML imports)
"""

import torch
import torch.nn as nn
import numpy as np
from AE.encoder import Encoder
from AE.Utils import *
from AE.decoder import Decoder
from config import *

class VQVAE(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers+1, res_h_dim)
        self.post_decode_c = nn.Conv2d(92, 1, kernel_size=2, stride=1, padding=1)
        self.post_decode_l = nn.Linear(186, in_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e.unsqueeze(0))
        x_hat = self.decoder(z_q)
        x_hat = x_hat.squeeze(0).unsqueeze(-1)
        x_hat = self.post_decode_c(x_hat).flatten()
        x_hat = self.post_decode_l(x_hat)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
    
    def encode(self, X, verbose=False):
        z_e = self.encoder(X)
        z_e = self.pre_quantization_conv(z_e)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            assert False
        return z_e