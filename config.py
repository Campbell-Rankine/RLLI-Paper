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
    'save path': 'C:/Code/RLLI-Paper/ae_checkpoints/',
    'additional transforms': [],
    'recon_weight': 1,
    'embedding_weight': 1,
}

trade_params = {
    'max_stock_value': 100000000,
    'initial_fund': np.random.randint(low=0, high=1, size=138),
}

general_params = {
    'ae_path': 'C:/Code/RLLI-Paper/vqvae_data_Auto_Encoder_',
    'ae_pt_epoch': str(4),
    'path': 'C:\Code\RLLI-Paper\dataset_long_ind.pickle',
    'debug_len': 50,
    'render_save': 'C:/Code/RLLI-Paper/Renders/',
    'drop_tickers': ['CEG'],
    'num_act': 3,
    'max_action': 5,
    'log dir': 'C:/Code/RLLI-Paper/results/maddpg_logging/'
}