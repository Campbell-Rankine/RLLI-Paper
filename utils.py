import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum

### - Main Loop Helper Functions - ###
def resume_train():
    raise NotImplementedError

def save_cp():
    raise NotImplementedError

def evalutaion_metrics(eval_data):
    raise NotImplementedError

class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long