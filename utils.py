import gym
from gym import spaces
from gym.utils import seeding
import torch as T
import numpy as np
from enum import Enum
from torchvision import transforms
from config import *
import pandas as pd


trans_list = [transforms.ToTensor(), 
                transforms.Normalize(0.5, 0.5)]


### - Main Loop Helper Functions - ###
def pre_input_process(X: pd.DataFrame, _transforms=[], head=True) -> T.tensor:
    if head:
        X = X.head(600)
        features = X.drop('close', axis=1).to_numpy()
    else:
        features = X
    _t = trans_list + _transforms
    _t = transforms.Compose([*_t])
    return _t(features)

import os
def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    T.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')

def resume_train():
    raise NotImplementedError

def save_cp():
    raise NotImplementedError

def evalutaion_metrics(eval_data):
    raise NotImplementedError

class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2


class Positions(Enum):
    Short = 0
    Long = 1
    Save = 2

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long
    
from config import *
import tqdm._tqdm
import argparse

from data import *

from Trader_MADDPG.MADDPG import *
from Trader_MADDPG.buffer import *
from AE.AE import *
from torch import optim

from torch.utils.tensorboard import SummaryWriter

def parse_args_main():
    import json

    parser = argparse.ArgumentParser()

    ### - Global Params - ###
    parser.add_argument("-debug", "--debug", dest="debug", metavar="debug", default = False,
                        type=bool, help="debug flag, minimize data to make things quicker to debug")

    parser.add_argument("-batch", "--batch", dest="batch", metavar="batch", default = 64,
                        type=int, help="default batch size")

    parser.add_argument("-e", "--e", dest="e", metavar="epochs", default = 64,
                        type=int, help="default num epochs")


    ### - AE Args - ###
    parser.add_argument("-aelr", "--aelr", dest="aelr", metavar="aelr", default = 0.1,
                        type=float, help="Auto Encoder Learning Rate")
    parser.add_argument("-aee", "--aee", dest="aee", metavar="aee", default = 64,
                        type=int, help="Auto Encoder Epochs")
    parser.add_argument("-aegc", "--aegc", dest="aegc", metavar="aegc", default = -1,
                        type=int, help="Gradient Norm Clipping Value")
    parser.add_argument("-aesv", "--aesv", dest="aesv", metavar="aesv", default = 1,
                        type=int, help="Gradient Norm Clipping Value")

    ### - Trading Args - ###
    parser.add_argument("-h1", "--h1", dest="h1", metavar="h1", default = 300,
                        type=int, help="1st Hidden Layer Size")
    
    parser.add_argument("-h2", "--h2", dest="h2", metavar="h2", default = 400,
                        type=int, help="2nd Hidden Layer Size")

    parser.add_argument("-lr", "--lr", dest="lr", metavar="lr", default = 0.1,
                        type=float, help="default learning rate")
    
    parser.add_argument("-a", "--a", dest="a", metavar="a", default = (0.000025, 0.99),
                        type=tuple, help="Network alphas")

    parser.add_argument("-b", "--b", dest="b", metavar="b", default = (0.00025, 0.99),
                        type=tuple, help="Network Betas")

    parser.add_argument("-g", "--g", dest="g", metavar="g", default = 0.99,
                        type=float, help="Network Gamma")

    parser.add_argument("-t", "--t", dest="t", metavar="t", default = 0.1,
                        type=float, help="Network Tau")

    parser.add_argument("-wd", "--wd", dest="wd", metavar="wd", default = 0.1,
                        type=float, help="Network Weight Decay")


    ### - Env Args - ###
    parser.add_argument("-render", "--render", dest="render", metavar="render", default = True,
                        type=bool, help="Gym Rendering Flag")

    ### - Misc. Args - ###
    parser.add_argument("-reward", "--reward", dest="reward", metavar="reward", default = 'base',
                        type=str, help="Reward input. See Doc File for reward function names, explainations, etc.")

    parser.add_argument("-verbose", "--verbose", dest="verbose", metavar="verbose", default = False,
                        type=bool, help="Print Env info")

    parser.add_argument("-ae", "--ae", dest="ae", metavar="ae", default = False,
                        type=bool, help='Latent space learning toggle')
    
    parser.add_argument("-mode", "--mode", dest="mode", metavar="mode", default = 'base',
                        type=str, help='Method of training: With or without autoencoded latent indicators')
    
    parser.add_argument("-loadae", "--loadae", dest="loadae", metavar="loadae", default = -1,
                        type=int, help='Method of training: With or without autoencoded latent indicators')
    
    ### - Bayes Opt Args - ###
    parser.add_argument("-acq", "--acq", dest="acq", metavar="acq", default = 'EI',
                        type=str, help='Acquisition Function for Bayesian Optimization')
    parser.add_argument("-BT", "--BT", dest="BT", metavar="BT", default = False,
                        type=bool, help='Toggle Bayes Opt')
    
    args = parser.parse_args()

    return args

class WMA():
    """
    Return a windowed moving average of data converted to a size
    """

    def __init__(self, data: np.array, return_size: int):
        self.data = data
        self.return_size = return_size

    def convert(self, size: int):
        if self.return_size == 0:
            return self.convert_exact(size)
        elif self.return_size == -1:
            return self.convert_as(size)
        elif self.return_size == 1:
            return self.convert_padded(size)
        else:
            raise AttributeError('Invalid input: make sure convert is corresponds to the following: -1 -> convert as is, no padding, size is reduced. 0 -> convert to a list of the exact size of the original. 1 -> 0 pad the data for conversion')
    def convert_exact(self, size:int):
        start_index = int(np.ceil(size/2)) #middle ground of 7 is 4 etc.
        end_index = len(self.data) - start_index
        _data = np.array([np.mean(self.data[i-start_index:i+start_index]) for i in range(start_index, end_index)])
        for j in range(start_index):
            np.insert(_data, j, np.mean(self.data[0:j]))
            np.insert(_data, j, np.mean(_data[-(j+1):-1]))
        return _data