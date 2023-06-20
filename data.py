import pandas as pd
import numpy as np
from config import general_params
import pickle

def _valid_df(data, key):
        if data[key] is None:
            return False
        return True

def zero_pad(data: pd.DataFrame, keys: list, len: int, beginning=True):
    """
    0 pad the dataframe from the beginning as this is the more correct way of doing it and the windowed approach makes data handling complicated
    """
    raise NotImplementedError

def load_dataset(file, debug=False):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        keys = list(data.keys())
        for x in general_params['drop_tickers']:
            keys.remove(x)
        for i, x in enumerate(keys):
            if not _valid_df(data, x):
                keys.pop(i)
        if debug:
            print(0., len(keys), general_params['debug_len'])
            inds = np.random.uniform(0., len(keys), general_params['debug_len'])
            inds = [int(x) for x in inds]
            keys = [keys[ind] for ind in inds] #Debug flag application
    return data, keys