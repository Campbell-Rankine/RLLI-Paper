import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rewards import *
from utils import *
import torch.nn.functional as F
from config import *

import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter


#TODO: Change this class to a dictionary of dataframes, iterating over multiple stocks

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, window_size, key, starting_funds=10000, in_house=.2, owned=0, shap=False, ae=None):
        self.df = df[key].drop('ticker', axis=1)
        assert self.df.ndim == 2

        self.r_fn = reward_1
        self.seed()
        self.name = 'env_' + key
        self.key = key
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        ### - Starting with some stock already in the inventory? - ###
        self.num_owned_0 = owned
        self.num_owned = owned

        self.ae = ae

        # profit/reward
        self.in_house = in_house

        self.starting_funds = starting_funds
        self.total_funds = starting_funds
        self.available_funds = (starting_funds) - (self.in_house * starting_funds) #Available funds to spend

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        self.init_episode()

    ### - Initializer - ###
    def init_episode(self):
        self.profit = 0.
        self.reward = 0.
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 2
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None


    ### - Main Gym Functions - ###
    def reset(self):
        self.net_worth = 0
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        self.num_owned = self.num_owned_0
        self.total_funds = self.starting_funds
        self.profit = 0.
        self.reward = 0.
        self.available_funds = (self.starting_funds) - (self.in_house * self.starting_funds)
        self.profit_0 = (self.prices[0]*self.num_owned_0) + self.available_funds
        return self._get_observation()


    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick >= self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False #Buy low sell high
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info


    ### - Utils - ###
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def display_config(self, verbose, obs=None):
        """
        Display the main characteristics of the environment, if calling at the beginning of training
        this function will call reset and return the observation for ease of use

        Args:
            obs:            (numpy.NDArray - (float)) - Most recent observable model input
        """
        if obs is None:
            obs = self.reset()
        if verbose:
            print(obs)
            print('Ticker: %s' % self.key)
            print('------------------------------------------')
            print('Observation Shape: ' + str(obs.shape))
            print('Reward Function: %s' % self.r_fn.__name__)
            print('Starting Funds: %.2f' % self.starting_funds)
            print('Available Funds %.2f' % self.available_funds)
            print('Timestep: %d' % self._current_tick)
            print('Number of Stock Owned: %d' % self.num_owned)

        return obs

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.profit
        )

        plt.pause(0.01)

    def render_all(self, mode='human'):
        tbwriter = SummaryWriter(general_params['log dir'])
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.profit)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.profit
        )
        
    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()
    

    ### - Helper Functions - ###
    def _process_data(self):
        prices = self.df['close'].to_numpy()
        features = self.df.drop('close', axis=1).to_numpy()
        return prices, features
    
    def _calculate_reward(self, action):
        return self.r_fn(self.available_funds, self.starting_funds, action, self.num_owned, self.prices, self._current_tick, self._end_tick)
    
    def _get_observation(self):
        if not self.ae is None:
            return self.latent_observation()
        else:
            obs = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]
            return obs

    def latent_observation(self):
        obs = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]
        X = pre_input_process(obs, head=False)
        X = F.normalize(X)
        return self.ae.encode(X.float()).detach().numpy()
    
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _update_profit(self, action):
        if action == 0:
            return None
        if action == -1 and not self.num_owned <= 0:
            self.num_owned -= 1
            self.available_funds += self.prices[self._current_tick]
        elif self.available_funds - self.prices[self._current_tick] >= 0:
            self.num_owned += 1
            self.available_funds -= self.prices[self._current_tick]
        if self._current_tick == self._start_tick:
            self.profit = 1
        self.net_worth = ((self.num_owned * self.prices[self._current_tick]) + self.available_funds)
        self.profit = self.net_worth - self.profit_0
        if self._current_tick % 100 == 0:
            self.profit_0 = (self.num_owned*self.prices[self._current_tick]) + self.available_funds
        self._total_profit += self.profit