### - ML IMPORTS - ###
import torch as T
import torch.nn as nn
import torchvision
from torch import optim

### - Other Library Imports - ###
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tqdm

### - Module Imports - ###
from AE.encoder import Encoder
from AE.Utils import *
from AE.AE import *
from AE.decoder import Decoder
from config import *
from data import *
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
    epochs = ae_params['epochs']
    print('Num Stocks: %d' % len(keys))
    shape = data[keys[0]].shape

    ### - Initialize Model - ###
    vqae = VQVAE(shape[1]-1, num_hidden, num_res_hid, num_res_lay, num_embeddings, latent, beta)
    optimizer = optim.Adam(vqae.parameters(), lr=lr, amsgrad=True)

    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
    }

    timesteps = list(range(0, shape[0] - (shape[0] % window)))
    print('Window Size: %i, length data: %i' % (window, shape[0] - (shape[0] % window)))

    ### - Train set reduction as this is meant to just be a pretrain - ###
    inds = np.random.uniform(0., len(keys), 100)
    inds = [int(x) for x in inds]
    keys = [keys[ind] for ind in inds] #Debug flag application

    ### - Train Loop - ###
    vqae.train()
    databar = tqdm(range(epochs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_var = np.var(np.vstack([F.normalize(T.tensor(data[x].values)) for x in keys]))
    for epoch_ in databar:
        epoch_losses = []
        epoch_perplexities = []
        epoch_recon_errors = []
        for key in keys:
            for i in timesteps:
                X = pre_input_process(data[key][i:i+window])
                X = F.normalize(X)
                ### - final initializations/steps - ###
                X.to(device)
                optimizer.zero_grad()

                ### - Forward Pass - ###
                recon_loss = 0
                #try: #Try - Except loop is bad practice but it's just an easier way of handling the mismatched data, because this is pretty much how its handled in the env
                try:
                    embedding_loss, x_hat, perplexity = vqae(X.float())
                except RuntimeError:
                    continue
                recon_loss = 0.
                if ae_params['recon_weight'] > 0.:
                    recon_loss = torch.mean((x_hat - X)**2) / train_var
                loss = (ae_params['recon_weight']*recon_loss) + (ae_params['embedding_weight']*embedding_loss)

                epoch_losses.append(loss.item())
                epoch_recon_errors.append(recon_loss.item())
                epoch_perplexities.append(perplexity.item())

                loss.backward()
                if args.aegc > 0:
                    torch.nn.utils.clip_grad_norm(vqae.parameters(), args.aegc)
                optimizer.step()
                databar.set_description('Epoch: %i, Key: %s, Epoch Loss %.5f, Epoch Recon Error: %.5f, Epoch Perplexity: %.2f' %  (epoch_, key, np.mean(epoch_losses), np.mean(epoch_recon_errors), np.mean(epoch_perplexities)))

        results['recon_errors'].append(np.mean(epoch_recon_errors))
        results['loss_vals'].append(np.mean(epoch_losses))
        results['perplexities'].append(np.mean(epoch_perplexities))

        if epoch_ % args.aesv == 0:
            results['n_updates'] = timesteps[-1] * len(keys) * epochs
            hyperparameters = ae_params
            save_model_and_results(vqae, results, hyperparameters, 'Auto_Encoder_' + str(epoch_))

            plt.scatter(list(range(len(results['recon_errors']))), results['recon_errors'], c='tab:blue')
            plt.savefig(ae_params['save path'] + 'recon_' + str(epoch_) + '.png')
            plt.clf()

            plt.scatter(list(range(len(results['loss_vals']))), results['loss_vals'], c='tab:pink')
            plt.savefig(ae_params['save path'] + 'loss_' + str(epoch_) + '.png')
            plt.clf()