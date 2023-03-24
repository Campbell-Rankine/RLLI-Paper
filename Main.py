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


from Data.data_utils import *
from Data.data import *

from Env import *
from rewards import *
from utils import *
from AE.PreTrain import train_ae

from rewards import *
from config import *
from Train import _valid_df, parse_args_main, base_train, latent_train, process_command_line_arguments

if __name__ == '__main__':
    args = parse_args_main()
    if args.ae:
        ### - Load Data - ###
        data, keys = load_dataset(general_params['path'], args.debug)
        train_ae(args, data, keys)
    else:
        base_train(args)


    