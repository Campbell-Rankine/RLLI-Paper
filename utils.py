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
def pre_input_process(X: pd.DataFrame, _transforms=[]) -> T.tensor:
    X = X.head(600)
    features = X.drop('close', axis=1).to_numpy()
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