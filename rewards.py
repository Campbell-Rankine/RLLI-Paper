### - Any Custom Built Reward Functions Go Here - ###
from utils import *

def reward_1(action, owned, prices, tick, avail, worth_0, discount=0.9):
    """
    Normalized 'worth' of current owned at the next timestep
    """
    diff = 0.
    opp = 0.
    if action == 1: 
        diff = 1
    elif action == -1:
        diff = -1
    else:
        return (owned*prices[tick+1]) / np.max(prices)

    curr_own = owned + diff
    return (curr_own + prices[tick+1]) / np.max(prices)

def reward_2(action, owned, prices, tick, avail, discount=0.9):
    """
    Normalized 'worth' averaged over discounted timesteps
    """
    diff = 0.
    opp = 0.
    if action == 1: 
        diff = 1
    elif action == -1:
        diff = -1
    else:
        return (owned*prices[tick+1]) / np.max(prices)
    curr_own = owned + diff
    rew = np.mean([discount**(i)*x for i, x in enumerate(prices)])
    return (curr_own + prices[tick+1]) / np.max(prices)

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