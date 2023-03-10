import torch as T
import torch.nn as nn
import numpy as np
import tqdm._tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms

from AE.AE import *
from AE.network_utils import *
from config import *

from utils import *
import pickle
import os
from tqdm import tqdm
import random

from Train import load_dataset, _valid_df

def static_train(args, epochs, device, test=0.1):

    dataset_p = general_params['path']

    lr = ae_params['lr']
    epochs = epochs
    optim_args = (lr, (0, 0.99))
    

    reg_fn = Regularizer(0.05)
    loss = get_loss_fn(args.loss)
    print(loss)
    assert(callable(loss_fn))

    ### - Load Data - ###
    data = load_dataset(general_params['path'])
    keys = list(data.keys())
    for x in general_params['drop_tickers']:
        keys.remove(x)
    for i, x in enumerate(keys):
        if not _valid_df(data, x):
            keys.pop(i)

    transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = StockData(dataset_p, ae_params['window'], device, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    num_features = dataset.data.shape[1]
    
    ### - Model Definition - ###
    encoder_args = {'batch_size': ae_params['batch'], 'window_size': ae_params['window'], 'latent': ae_params['latent'],
                    'dims': num_features}
    decoder_args = {}

    model = VGG16_AE(encoder_args, decoder_args, device)
    optim = T.optim.Adam(model.parameters(), *optim_args)
    scheduler = T.optim.lr_scheduler.ExponentialLR(optim, 0.9, last_epoch=- 1, verbose=False)

    model.train()
    databar = tqdm(range(epochs))
    epoch_losses = []

    ### - Penalty - ###
    g_pen = 0.
    penalty = 0.
    testing = []
    for epoch in databar[:ae_params['window']]:
        
        ### - Databar Init - ###
        losses = []
        databar.set_description('Epoch: %i, Loss: %0.2f, Grad: %0.2f, Regularization Penalty: %.2f, Sample #: %i' % 
        (epoch, 0., 0., 0., 0))
        running_loss = 0.0
        instances = 0
        
        ### - iterate through dataset - ###
        
        for i, x in enumerate(dataloader):
            if x is None:
                continue
            x.requires_grad = True
            if x.shape[1] < args.window:
                continue
            x = x.to(device)
            #model.zero_grad()
            optim.zero_grad()
            out, encoded_dims = model(x.detach())
            print(out.shape, x.shape)
            loss_ = loss(out, x)
            try:
                model.eval()
                p = T.norm(model.encoder.get_activations_gradient(), 'fro')
                model.train()
            except:
                p = 0.
            loss_ += p
            loss_.backward()
            
            #losses.append(loss_.item())
            running_loss += np.abs(loss_.item()) / args.batch
            
            databar.set_description('Epoch: %i, Loss: %0.2f, Running Loss: %.2f, Grad Penalty: %e, Sample #: %i, Encoded Dims: %i' % 
                                    (epoch, loss_.item(), running_loss, p, i, encoded_dims[1]))
            nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optim.step()
            scheduler.step()
        epoch_losses.append(np.mean(losses))

    print('Saving Trained Network')
    check_point = { 'model' : model.state_dict(), 
                    'optimizer' : optim.state_dict(),
                    'epoch_losses' : epoch_losses,
                           }
    T.save(check_point, 'C:\Code\RLLI-Paper\SM_Representation_Net.pth')
    print('Done!')