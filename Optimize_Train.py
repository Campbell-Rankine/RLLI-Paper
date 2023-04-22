### - Misc imports - ###
from config import *
import tqdm._tqdm
import argparse

### - Data Imports (Deprecated TODO: Get rid of these imports) - ###
from Data.data import *
from Data.data_utils import *

### - Library imports - ###
from Trader_MADDPG.MADDPG import *
from Trader_MADDPG.buffer import *
from AE.AE import *
from torch import optim

### - Logging - ###
from torch.utils.tensorboard import SummaryWriter

### - Bayes Opt. - ###
from Optimization.Acquisition_Functions import *
from Optimization.Bayesian_Optimization import *

### - Config Declarations - ###
lr = ae_params['lr']
beta = ae_params['beta']
num_updates = ae_params['num_updates']
num_hidden = ae_params['num_hiddens']
num_res_hid = ae_params['num_residual_hiddens']
num_res_lay = ae_params['num_residual_layers']
log_interval = ae_params['log_interval']
epochs = ae_params['epochs']
latent = ae_params['latent']
window = ae_params['window']
save_path = ae_params['save path']
additional_transforms = ae_params['additional transforms']
num_embeddings = ae_params['num embeddings']

def obj_func(test_data):
    raise NotImplementedError

def bayesian_train(args, data, keys):

    raise NotImplementedError