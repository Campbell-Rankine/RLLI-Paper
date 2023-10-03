#Config file for multiple networks. Standard python dict
import numpy as np

ae_params = {
    'lr': 3e-4,
    'beta': 0.25,
    'num_updates': 5000,
    'num_hiddens': 128,
    'num_residual_hiddens': 64,
    'num_residual_layers': 3,
    'log_interval': 50,
    'epochs': 10,
    'batch': 1,
    'window': 30,
    'latent': 64,
    'num embeddings': 512,
    'save path': 'ae_checkpoints/',
    'additional transforms': [],
    'recon_weight': 1,
    'embedding_weight': 1,
}

trade_params = {
    'max_stock_value': 100000000,
    'initial_fund': np.random.randint(low=0, high=1, size=138),
}

general_params = {
    'ae_path': 'vqvae_data_Auto_Encoder_',
    'ae_pt_epoch': str(4),
    'path': 'data.pickle',
    'debug_len': 50,
    'render_save': '/home/campbell/Desktop/Python-Projects/RLLI-Paper/Renders/',
    'drop_tickers': ['CEG'],
    'num_act': 3,
    'max_action': 5,
    'log dir': '/home/campbell/Desktop/Python-Projects/RLLI-Paper/Results/logging/',
    'test_indices': [540, 620],
    'drop_keys': ['OGN']
}

env_params = {
    'plot': 5,
    'plot_list': None,
}