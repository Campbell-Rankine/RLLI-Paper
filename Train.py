from config import *
import tqdm._tqdm
import argparse

from Data.data import *
from Data.data_utils import *

from Trader_MADDPG.MADDPG import *
from Trader_MADDPG.buffer import *
from AE.AE import *
from torch import optim

from torch.utils.tensorboard import SummaryWriter

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

def parse_args_main():
    import json

    parser = argparse.ArgumentParser()

    ### - Global Params - ###
    parser.add_argument("-debug", "--debug", dest="debug", metavar="debug", default = False,
                        type=bool, help="debug flag, minimize data to make things quicker to debug")

    parser.add_argument("-batch", "--batch", dest="batch", metavar="batch", default = 64,
                        type=int, help="default batch size")

    parser.add_argument("-e", "--e", dest="e", metavar="epochs", default = 64,
                        type=int, help="default num epochs")


    ### - AE Args - ###
    parser.add_argument("-aelr", "--aelr", dest="aelr", metavar="aelr", default = 0.1,
                        type=float, help="Auto Encoder Learning Rate")
    parser.add_argument("-aee", "--aee", dest="aee", metavar="aee", default = 64,
                        type=int, help="Auto Encoder Epochs")
    parser.add_argument("-aegc", "--aegc", dest="aegc", metavar="aegc", default = -1,
                        type=int, help="Gradient Norm Clipping Value")
    parser.add_argument("-aesv", "--aesv", dest="aesv", metavar="aesv", default = 1,
                        type=int, help="Gradient Norm Clipping Value")

    ### - Trading Args - ###
    parser.add_argument("-h1", "--h1", dest="h1", metavar="h1", default = 300,
                        type=int, help="1st Hidden Layer Size")
    
    parser.add_argument("-h2", "--h2", dest="h2", metavar="h2", default = 400,
                        type=int, help="2nd Hidden Layer Size")

    parser.add_argument("-lr", "--lr", dest="lr", metavar="lr", default = 0.1,
                        type=float, help="default learning rate")
    
    parser.add_argument("-a", "--a", dest="a", metavar="a", default = (0.000025, 0.99),
                        type=tuple, help="Network alphas")

    parser.add_argument("-b", "--b", dest="b", metavar="b", default = (0.00025, 0.99),
                        type=tuple, help="Network Betas")

    parser.add_argument("-g", "--g", dest="g", metavar="g", default = 0.99,
                        type=float, help="Network Gamma")

    parser.add_argument("-t", "--t", dest="t", metavar="t", default = 0.1,
                        type=float, help="Network Tau")

    parser.add_argument("-wd", "--wd", dest="wd", metavar="wd", default = 0.1,
                        type=float, help="Network Weight Decay")


    ### - Env Args - ###
    parser.add_argument("-render", "--render", dest="render", metavar="render", default = False,
                        type=bool, help="Gym Rendering Flag")

    ### - Misc. Args - ###
    parser.add_argument("-reward", "--reward", dest="reward", metavar="reward", default = 'base',
                        type=str, help="Reward input. See Doc File for reward function names, explainations, etc.")

    parser.add_argument("-verbose", "--verbose", dest="verbose", metavar="verbose", default = False,
                        type=bool, help="Print Env info")

    parser.add_argument("-ae", "--ae", dest="ae", metavar="ae", default = False,
                        type=bool, help='Latent space learning toggle')
    
    parser.add_argument("-mode", "--mode", dest="mode", metavar="mode", default = 'base',
                        type=str, help='Method of training: With or without autoencoded latent indicators')
    
    parser.add_argument("-loadae", "--loadae", dest="loadae", metavar="loadae", default = -1,
                        type=int, help='Method of training: With or without autoencoded latent indicators')
    
    args = parser.parse_args()

    return args

def _valid_df(data, key):
        if data[key] is None:
            return False
        return True

def base_train(args, data, keys):
    print('Begin Training')

    ### - Load Data - ###
    epochs = args.e
    if args.debug:
        epochs = 500

    print('Num Stocks: %d' % len(keys))

    ### - Create Models - ###
    env_args = { #Environments args to be passed to each agent
        'df': data,
        'window_size': 1,
        'key': keys,
        'rew_fn': args.reward,
    }

    bots = MADDPG(134, len(keys) * 134, keys, general_params['num_act'], env_args, args.verbose) #init bots
    mem = MultiAgentReplayBuffer(100000, len(keys) * 134, 134, general_params['num_act'], len(keys), 1)

    ### - START TRAIN LOOP - ###
    total_steps = 0 #Logging vars
    total_score = 0
    databar = tqdm(range(epochs+1))
    actionwriter = SummaryWriter(general_params['log dir'] + 'actions')
    for i in databar: #Start of main loop
         ### - Epoch Setup - ###
        bots.reset_environments(i, mem) #Basic loggers and trackers
        score = 0
        dones = [False]*bots.n_agents
        episode_steps = 0

        ### - TensorBoard Logging - ###

        while not any(dones):
            ### - Episode Steps - ###
            score, total_steps, episode_steps, infos, _dones, probs, actions = bots.step(mem, total_steps, episode_steps, score) #Run Envs
            total_score+= score
            dones = _dones
            if episode_steps % 50 ==0:
                databar.set_description('Epoch %d, Current Iters: %d, Episode Iters: %d, Mean Owned: %.2f. Mean Profit: %.2f, Mean Funds: %.2f, Sum Profit: %.2f' % 
                (i, total_steps, episode_steps, sum([x.env.num_owned for x in bots.agents]) / bots.n_agents, sum([x.env.profit for x in bots.agents]) / bots.n_agents, 
                sum([x.env.available_funds for x in bots.agents]) / len(bots.agents), sum([x.env.profit for x in bots.agents]) )) #Logging
            actionwriter.add_scalar('total_profit_' + str(i), sum([x.env.profit for x in bots.agents]), total_steps)
            actionwriter.flush()
            for j, x in enumerate(bots.agents):
                actionwriter.add_scalar(x.name + '_action', actions[j], total_steps)
                actionwriter.add_scalars(x.name + '_profit', {'profit': x.env.profit, 'price': x.env.prices[x.env._current_tick]}, total_steps)
            actionwriter.add_scalars('mean_losses_loss', {'critic loss': bots.critic_loss, 'actor loss': bots.actor_loss, 'encoder loss': bots.encoder_loss}, total_steps)
            actionwriter.flush()
        if i % 50 == 0:
            bots.get_renders(i, keys)
    ### - Model Save - ###
    actionwriter.close()
    bots.save_checkpoint()
    ### - Evaluation - ###

def latent_train(args, data, keys):
    print('Begin Latent Training')

    ### - Load Data - ###
    keys = list(data.keys())
    for x in general_params['drop_tickers']:
        keys.remove(x)
    for i, x in enumerate(keys):
        if not _valid_df(data, x):
            keys.pop(i)
    epochs = args.e

    ### - Set Debug - ###
    if args.debug:
        print(0., len(keys), general_params['debug_len'])
        inds = np.random.uniform(0., len(keys), general_params['debug_len'])
        inds = [int(x) for x in inds]
        keys = [keys[ind] for ind in inds] #Debug flag application
        epochs = 500

    print('Num Stocks: %d' % len(keys))
    ### - Load AutoEncoder - ###
    ae_path = general_params['ae_path'] + general_params['ae_pt_epoch'] + '.pth'
    print('loading Auto Encoder from path: %s' % ae_path)
    shape = data[keys[0]].shape
    vqae = VQVAE(shape[1]-1, num_hidden, num_res_hid, num_res_lay, num_embeddings, latent, beta)
    if args.loadae > 0:
        ae_path = general_params['ae_path'] + str(args.loadae) + '.pth'
        cp = T.load(ae_path)
        vqae.load_state_dict(cp['model'])
    ae_optimizer = optim.Adam(vqae.encoder.parameters(), lr=lr, amsgrad=True)
    print(vqae)

    ### - Create Models - ###
    env_args = { #Environments args to be passed to each agent
        'df': data,
        'window_size': ae_params['window'],
        'key': keys,
        'rew_fn': args.reward,
        'ae': vqae,
    }

    bots = MADDPG(latent, len(keys) * latent, keys, 3, env_args, args.verbose, latent=True, latent_optimizer=ae_optimizer) #init bots
    mem = MultiAgentReplayBuffer(100000, len(keys) * latent, latent, 3, len(keys), 1)

    ### - START TRAIN LOOP - ###
    total_steps = 0 #Logging vars
    total_score = 0
    databar = tqdm(range(epochs+1))

    actionwriter = SummaryWriter(general_params['log dir'] + 'actions')

    for i in databar: #Start of main loop
         ### - Epoch Setup - ###
        bots.reset_environments(i, mem) #Basic loggers and trackers
        score = 0
        dones = [False]*bots.n_agents
        episode_steps = 0

        ### - TensorBoard Logging - ###

        while not any(dones):
            ### - Episode Steps - ###
            score, total_steps, episode_steps, infos, _dones, probs, actions = bots.step(mem, total_steps, episode_steps, score) #Run Envs
            total_score+= score
            dones = _dones
            if episode_steps % 50 ==0:
                databar.set_description('Epoch %d, Current Iters: %d, Episode Iters: %d, Mean Owned: %.2f. Mean Profit: %.2f, Mean Funds: %.2f, Sum Profit: %.2f' % 
                (i, total_steps, episode_steps, sum([x.env.num_owned for x in bots.agents]) / bots.n_agents, sum([x.env.profit for x in bots.agents]) / bots.n_agents, 
                sum([x.env.available_funds for x in bots.agents]) / len(bots.agents), sum([x.env.profit for x in bots.agents]) )) #Logging
            actionwriter.add_scalar('total_profit_' + str(i), sum([x.env._total_profit for x in bots.agents]), total_steps)
            actionwriter.add_scalar('net_worth_' + str(i), sum([x.env.net_worth for x in bots.agents]), total_steps)
            actionwriter.flush()
            for j, x in enumerate(bots.agents):
                actionwriter.add_scalar(x.name + '_action', actions[j], total_steps)
                actionwriter.add_scalars(x.name + '_profit', {'profit': x.env.profit, 'price': x.env.prices[x.env._current_tick]}, total_steps)
            actionwriter.add_scalars('mean_losses_loss', {'critic loss': bots.critic_loss, 'actor loss': bots.actor_loss, 'encoder loss': bots.encoder_loss}, total_steps)
            actionwriter.flush()
        if i % 50 == 0:
            bots.get_renders(i, keys)
            
    ### - Model Save - ###
    bots.save_checkpoint()
    actionwriter.close()
    ### - Evaluation - ###