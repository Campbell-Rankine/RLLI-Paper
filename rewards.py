### - Any Custom Built Reward Functions Go Here - ###
from utils import *

def reward_1(action, owned, prices, tick, avail, worth_0):
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
    elif action == -1:
        diff = -1
        opp = 1
    else:
        a_prof = (((owned) * prices[tick]) + (avail + prices[tick]))
        o_prof = np.mean((((owned + -1) * prices[tick]) + (avail + -1*prices[tick])), (((owned + 1) * prices[tick]) + (avail + 1*prices[tick])))
        max_p = np.max([a_prof, o_prof])
        return (((a_prof - o_prof) / abs(max_p))) - 1

    a_prof = (((owned + diff) * prices[tick]) + (avail + diff*prices[tick])) - worth_0
    o_prof = (((owned + opp) * prices[tick]) + (avail + opp*prices[tick])) - worth_0

    max_p = np.max([a_prof, o_prof])
    if max_p == 0.:
        return 0.
    if a_prof == 0 and o_prof == 0:
        return 0.
    return -(((a_prof - o_prof) / abs(max_p))) - 1

def reward_2(action, owned, prices, tick, avail, worth_0):
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
    elif action == -1:
        diff = -1
        opp = 1
    else:
        a_prof = (((owned) * prices[tick]) + (avail + prices[tick])) - worth_0
        O_1 = (((owned + -1) * prices[tick]) + (avail + -1*prices[tick])) - worth_0
        O_2 = (((owned + 1) * prices[tick]) + (avail + 1*prices[tick])) - worth_0
        o_prof = np.mean([O_1, O_2])
        max_p = np.max([a_prof, o_prof])
        return (((a_prof - o_prof) / abs(max_p))) - 1

    ### - Profit - ###
    a_prof = (((owned + diff) * prices[tick]) + (avail + diff*prices[tick])) - worth_0
    o_prof = (((owned + opp) * prices[tick]) + (avail + opp*prices[tick])) - worth_0

    max_p = np.max([a_prof, o_prof])
    if max_p == 0.:
        return 0.
    if a_prof == 0 and o_prof == 0:
        return 0.
    return (((a_prof - o_prof) / abs(max_p))) - 1

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