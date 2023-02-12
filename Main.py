### - Standard Imports - ###
import torch as T
import torch.nn as nn
import numpy as np
import pandas as pd
import collections
import ray

from Trader.network import *
from Trader.utils import *
from Trader.buffer import *

from Data.data import *
from Data.data_utils import *

from AE.AE import *
from AE.network_utils import *

from rewards import *

if __name__ == '__main__':
    print('Begin Training')

    data = load_dataset('dataset_long_ohlc.pickle')
    keys = list(data.keys())
    print(data[keys[0]])

    