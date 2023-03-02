### - Standard Imports - ###
import torch as T
import torch.nn as nn
import numpy as np
import pandas as pd
import collections
import ray

from Trader_MADDPG.buffer import *
from Trader_MADDPG.MADDPG import *
from Trader_MADDPG.network import *
from Trader_MADDPG.utils import *


from Data.data_utils import *
from Data.data import *

from Env import *
from rewards import *
from utils import *

from rewards import *
from config import *

def parse_args():
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
                        
    parser.add_argument("-loss", "--loss", dest="loss", metavar="loss", default = 'BCE',
                        type=str, help="Loss for reconstruction objective")

    parser.add_argument("-latent", "--latent", dest="latent", metavar="latent", default = 138,
                        type=int, help="latent size")

    parser.add_argument("-window", "--window", dest="window", metavar="window", default = 30,
                        type=int, help="default window training size")

    parser.add_argument("-sched", "--sched", dest="sched", metavar="sched", default = False,
                        type=bool, help="Lr scheduler switch")


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
    parser.add_argument("-render", "--render", dest="render", metavar="render", default = False,
                        type=bool, help="Gym Rendering Flag")

    ### - Misc. Args - ###
    parser.add_argument("-reward", "--reward", dest="reward", metavar="reward", default = 'base',
                        type=str, help="Reward input. See Doc File for reward function names, explainations, etc.")
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    print('Begin Training')
    

    args = parse_args()

    """### - Load Data - ###
    data = load_dataset(general_params['path'])
    keys = list(data.keys())

    if args.debug:
        keys = keys[:general_params['debug_len']] #Debug flag application

    ### - Create Models - ###
    print('Create AE Model')
    #TODO: Maybe try and get the base RL class working first. Then add and tweak the AE

    print('Create Trader')
    #TODO: Make sure with new training structure that this input shape will work over all key lists

    print('Create Env')
    reward = get_rew(args.reward) #get reward function
    assert(callable(reward))
    """
    ### - Basic inits for the project - ###
    data = load_dataset('C:\Code\RLLI-Paper\dataset_long_ind.pickle')
    keys = list(data.keys())[0:10] #Useful in the experiments tab to keep everything a little smaller

    env_args = { #Environments args to be passed to each agent
        'df': data,
        'window_size': 1,
        'key': keys,
    }

    bots = MADDPG(134, len(keys) * 134, keys, 2, env_args) #init bots

    mem = MultiAgentReplayBuffer(1000000, len(keys) * 134, 134, 2, len(keys), 1)


    score, total_steps, episode_steps, infos, _dones = bots.step(mem, 0, 0)

    ### - Begin Loops - ###

    ### - Evaluation - ###



    