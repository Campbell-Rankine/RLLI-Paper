#Config file for multiple networks. Standard python dict
import numpy as np

ae_params = {
    'debug_len': 10,
}

trade_params = {
    'max_stock_value': 100000000,
    'initial_fund': np.random.randint(low=0, high=1, size=138),
}

general_params = {
    'dataset_path': None,
}