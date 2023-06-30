import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from config import *
from utils import WMA
import matplotlib.pyplot as plt


def test_additional_input_functions(am_func):
    if not callable(am_func) and not am_func is None:
        assert(type(am_func) == list) # Assert list typing for am_func as you can either pass a function or a list of functions (TODO)
        for x in am_func:
            assert(callable(x))

def test_all_bots(bots, mem, epoch, logging=True, am_func=None):
    """
    Run a testing evaluation on the bots using the Env Testing window to allow for easier day to day decision making

    Args:
        bots (MADDPG - Custom Class) - Collection of RL agents for each stock
        am_func (callable) - Additional Metric functions to add to the evaluation. User defined and should take the outputs defined by the trading bots
    """
    """
    Testing scalars for Summary Writer
    
            actionwriter.flush()
            for j, x in enumerate(bots.agents):
                actionwriter.add_scalar(x.name + '_action', actions[j], total_steps)
                actionwriter.add_scalars(x.name + '_profit', {'profit': x.env.profit, 'price': x.env.prices[x.env._current_tick]}, total_steps)
    """
    test_additional_input_functions(am_func)

    ### - Initialize Trackers - ###
    test_score = 0
    test_steps = 0

    ### - Start environments - ###
    bots.reset_environments(0, mem, test=True) #need to call a reset environments for testing environments
    dones = [False]*bots.n_agents

    ### - Summary Writer - ###
    actionwriter = SummaryWriter(general_params['log dir'] + 'actions')
    test_scores = []
    while not any(dones):
        test_score, test_steps, test_steps, infos, _dones, probs, actions = bots.step_eval(mem, test_steps, test_steps, test_score) #Run Envs
        test_score += sum([x.test_env._total_profit for x in bots.agents])
        dones = _dones
        for j, x in enumerate(bots.agents):
                if j % 15 == 0:
                    #actionwriter.add_scalars(x.name + '_action', actions[j], test_steps)
                    actionwriter.add_scalar(x.name + '_price', x.test_env.prices[x.test_env._current_tick], test_steps)
                    actionwriter.add_scalar(x.name + '_worth', x.test_env.net_worth, test_steps)
        test_scores.append(sum([x.test_env._total_profit for x in bots.agents]))
    actionwriter.flush()
    mov_avg = WMA(test_scores, 0)
    test_scores = mov_avg.convert(3)
    plt.plot(range(len(test_scores)), test_scores)
    plt.savefig(general_params['render_save'] + 'test_score_ema_' + str(epoch) + '.png')
    plt.clf()
    print('Testing total profit was: %0.2f' % test_score)
    return test_score