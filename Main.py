### - Standard Imports - ###
import torch as T
import torch.nn as nn
import numpy as np
import pandas as pd
import collections
import ray
import random

from Trader_MADDPG.buffer import *
from Trader_MADDPG.MADDPG import *
from Trader_MADDPG.network import *
from Trader_MADDPG.utils import *

from data import *

from Env import *
from rewards import *
from utils import *
from AE.PreTrain import train_ae

from rewards import *
from config import *
from utils import *

#TODO: change main function to include data

if __name__ == '__main__':
    args = parse_args_main()
    data, keys = load_dataset(general_params['path'], args.debug)
    keys = keys[:-20]
    if args.ae:
        ### - Load Data - ###
        train_ae(args, data, keys)
    elif args.mode == 'latent'and not args.BT:
        from TrainStructures.latent import *
        latent_train(args, data, keys)
    elif args.mode == 'latent'and args.BT:
        test_percentage = 0.1
        inds = np.random.uniform(0., len(keys), int(test_percentage*len(keys)))
        inds = [int(x) for x in inds]
        train_keys = [keys[ind] for ind in inds] #Debug flag application
        test_keys = list(set(keys)-set(train_keys))
        from TrainStructures.optimizer import *
        bayesian_train(args, data, train_keys, test_keys)
    else:
        from TrainStructures.base import *
        base_train(args, data, keys)


    