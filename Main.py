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

    parser.add_argument("-verbose", "--verbose", dest="verbose", metavar="verbose", default = False,
                        type=bool, help="Print Env info")
    
    args = parser.parse_args()

    return args

def _valid_df(data, key):
        if data[key] is None:
            return False
        return True

if __name__ == '__main__':
    print('Begin Training')
    

    args = parse_args()

    ### - Load Data - ###
    data = load_dataset(general_params['path'])
    keys = list(data.keys())
    for x in general_params['drop_tickers']:
        keys.remove(x)
    for i, x in enumerate(keys):
        if not _valid_df(data, x):
            keys.pop(i)
    epochs = args.e

    ### - Set Debug - ###
    if args.debug:
        print(0., len(keys), general_params['debug_len'])
        inds = np.random.uniform(0., len(keys), general_params['debug_len'])
        inds = [int(x) for x in inds]
        keys = [keys[ind] for ind in inds] #Debug flag application
        epochs = 500

    print('Num Stocks: %d' % len(keys))

    ### - Create Models - ###
    env_args = { #Environments args to be passed to each agent
        'df': data,
        'window_size': 1,
        'key': keys,
        'rew_fn': args.reward,
    }
    bots = MADDPG(134, len(keys) * 134, keys, 2, env_args, args.verbose) #init bots
    mem = MultiAgentReplayBuffer(100000, len(keys) * 134, 134, 2, len(keys), 1)

    ### - START TRAIN LOOP - ###
    total_steps = 0 #Logging vars
    total_score = 0
    databar = tqdm(range(epochs+1))

    for i in databar: #Start of main loop
        ### - Epoch Setup - ###
        bots.reset_environments() #Basic loggers and trackers
        score = 0
        dones = [False]*bots.n_agents
        episode_steps = 0

        while not any(dones):
            ### - Episode Steps - ###
            score, total_steps, episode_steps, infos, _dones = bots.step(mem, total_steps, episode_steps, score) #Run Envs
            total_score+= score
            dones = _dones
            if episode_steps % 100 ==0:
                databar.set_description('Epoch %d, Episode_Score: %.2f, Total Score: %.2f, Current Iters: %d, Episode Iters: %d, Mean Owned: %.2f. Mean Profit: %.2f, Mean Funds: %.2f' % 
                (i, score, total_score, total_steps, episode_steps, sum([x.env.num_owned for x in bots.agents]) / bots.n_agents, sum([x.env.profit for x in bots.agents]) / bots.n_agents, 
                sum([x.env.available_funds for x in bots.agents]) / len(bots.agents))) #Logging
        if i % 50 == 0:
            bots.get_renders(i)
    ### - Model Save - ###
    bots.save_checkpoint()
    ### - Evaluation - ###



    