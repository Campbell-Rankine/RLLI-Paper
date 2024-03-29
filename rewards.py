### - Any Custom Built Reward Functions Go Here - ###
from utils import *


### - Reward Function - ###
def reward_1(funds, starting_funds, action, owned, prices, tick, total_timesteps):
    """
    Normalized 'worth' of current owned at the next timestep. No testing support modified
    """
    diff = 0.
    opp = 0.
    if action == 1: 
        diff = 1
    elif action == -1:
        diff = -1
    else:
        return -((((owned*prices[tick+1]) / np.max(prices))) + (funds/starting_funds)) / total_timesteps # Change epochs to tick in the final version

    curr_own = owned + diff
    return -((((owned*prices[tick+1]) / np.max(prices))) + (funds/starting_funds)) / total_timesteps

def reward_2(action, owned, prices, tick, avail, discount=0.9):
    """
    Normalized 'worth' averaged over discounted timesteps

    TODO: Fix this to make it a time dependent reward. Raise discount to the t, and multiply the reward by that
    """
    
    return (discount**tick)*reward_1(action, owned, prices, tick, avail, 0)

def reward_3(action, owned, prices, tick, avail):
    """
    Simpler reward function simply defined by the total profit over all agents

    Args:
        action:             (float) - Action output from the network
        owned:              (int) - Number of this stock currently owned
        prices:             (numpy.NDArray - (float)) - Array of prices for all timesteps
        tick:               (int) - current timestep
        avail:              (float) - Available funds for calculating profit/net worth
    """
    diff = 0.
    opp = 0.
    if action == 1: 
        diff = 1
        opp = -1
    else:
        diff = -1
        opp = 1



    raise NotImplementedError

def get_rew(input: str):
    if input == 'base':
        return reward_1
    if input == 'base_profit':
        return reward_2
    else:
        raise ValueError