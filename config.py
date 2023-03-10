#Config file for multiple networks. Standard python dict
import numpy as np

ae_params = {
    'lr': 0.1,
    'epochs': 64,
    'loss': 'BCE',
    'batch': 1,
    'window': 1,
    'latent': 64,
}

trade_params = {
    'max_stock_value': 100000000,
    'initial_fund': np.random.randint(low=0, high=1, size=138),
}

general_params = {
    'path': 'C:\Code\RLLI-Paper\dataset_long_ind.pickle',
    'debug_len': 50,
    'render_save': 'C:/Code/RLLI-Paper/Renders/',
    'drop_tickers': ['CEG'],
    'num_act': 3,
    'max_action': 5
}