### - Any Custom Built Reward Functions Go Here - ###
from utils import *

def reward_1(action, owned, prices, tick, avail):
    """
    Reward function defined as: The profit obtained by taking this action, over the opposite action. 
    (profit_this - profit_opposite). Normalized by the max profit.

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

    ### - Profit - ###
    if avail - prices[tick] < 0 or owned + diff < 0:
        return -10
    a_prof = (((owned + diff) * prices[tick]) + (avail - prices[tick])) - ((owned * prices[tick]) + avail) #action profit
    if owned == 0:
        o_prof = 0
    else:
        o_prof = (((owned + opp) * prices[tick]) + (avail + prices[tick])) - ((owned * prices[tick]) + avail) #opposite

    max_p = np.max([a_prof, o_prof])
    if a_prof == 0 and o_prof == 0:
        return 0.
    return (a_prof - o_prof) / max_p

def reward_2(action, owned, prices, tick, avail):
    """
    Simpler reward function simply defined by the total profit over all agents

    Args:
        action:             (float) - Action output from the network
        owned:              (int) - Number of this stock currently owned
        prices:             (numpy.NDArray - (float)) - Array of prices for all timesteps
        tick:               (int) - current timestep
        avail:              (float) - Available funds for calculating profit/net worth
    """
    
    raise NotImplementedError

def get_rew(input: str):
    if input == 'base':
        return reward_1
    else:
        raise ValueError