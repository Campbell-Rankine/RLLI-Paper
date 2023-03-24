### - ML IMPORTS - ###
import torch as T
import torch.nn as nn
import torchvision
from torch import optim

### - Other Library Imports - ###
import numpy as np
import argparse

### - Module Imports - ###
from AE.encoder import Encoder
from AE.Utils import *
from AE.AE import *
from AE.decoder import Decoder
from config import *
from Data.data import *
from Train import _valid_df
from utils import pre_input_process
from utils import save_model_and_results

### - Config Declarations - ###
lr = ae_params['lr']
beta = ae_params['beta']
num_updates = ae_params['num_updates']
num_hidden = ae_params['num_hiddens']
num_res_hid = ae_params['num_residual_hiddens']
num_res_lay = ae_params['num_residual_layers']
log_interval = ae_params['log_interval']
epochs = ae_params['epochs']
latent = ae_params['latent']
window = ae_params['window']
save_path = ae_params['save path']
additional_transforms = ae_params['additional transforms']
num_embeddings = ae_params['num embeddings']

def train_ae(args, data, keys):
    ### - get args - ###
    print('Begin Training')
    epochs = args.aee
    print('Num Stocks: %d' % len(keys))

    ### - Initialize Model - ###
    vqae = VQVAE(num_hidden, num_res_hid, num_res_lay, num_embeddings, latent, beta)
    optimizer = optim.Adam(vqae.parameters(), lr=lr, amsgrad=True)

    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
    }

    shape = data[keys[0]].shape
    timesteps = list(range(0, shape[0] - (shape[0] % window)))
    print('Window Size: %i, length data: %i' % (window, shape[0] - (shape[0] % window)))

    ### - Train Loop - ###
    vqae.train()
    databar = tqdm(range(epochs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_var = np.var(np.vstack([data[x] for x in keys])) / np.max(np.vstack([data[x] for x in keys]))
    for epoch_ in databar:
        epoch_losses = []
        epoch_perplexities = []
        epoch_recon_errors = []
        for key in keys:
            for i in timesteps:
                X = pre_input_process(data[key][i:i+window])
                ### - final initializations/steps - ###
                X.to(device)
                optimizer.zero_grad()

                ### - Forward Pass - ###
                embedding_loss, x_hat, perplexity = vqae(X.float())
                x_hat = F.interpolate(x_hat, (134))
                recon_loss = 0
                try:
                    recon_loss = torch.mean((x_hat - X)**2) / train_var
                except RuntimeError:
                    continue
                loss = recon_loss + embedding_loss

                epoch_losses.append(loss.item())
                epoch_recon_errors.append(recon_loss.item())
                epoch_perplexities.append(perplexity.item())

                loss.backward()
                optimizer.step()

                databar.set_description('Epoch: %i, Key: %s, Epoch Loss %.2f, Epoch Recon Error: %.2f, Epoch Perplexity: %.2f' %  (epoch_, key, np.mean(epoch_losses), np.mean(epoch_recon_errors), np.mean(epoch_perplexities)))
        results['recon_errors'].append(np.mean(epoch_recon_errors))
        results['loss_vals'].append(np.mean(epoch_losses))
        results['perplexities'].append(np.mean(epoch_perplexities))
    
    ### - Final Updates and Save - ###
    results['n_updates'] = timesteps[-1] * len(keys) * epochs
    hyperparameters = ae_params
    save_model_and_results(vqae, results, hyperparameters, ae_params['save path'])