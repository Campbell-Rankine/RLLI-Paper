### - Misc imports - ###
from config import *
import tqdm._tqdm
import argparse

from data import *

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

def obj_func(test_data, bots, mem):
    """
    Run 1 Training episode on unseen data and return the average profit from each network as our metric. From here maximize this testing data with respect to the hyperparameters
    """
    while not any(dones):
        ### - Episode Steps - ###
        score, total_steps, episode_steps, infos, _dones, probs, actions = bots.step_eval(mem, total_steps, episode_steps, score) #Run Envs
        total_score+= score
        dones = _dones
    return score

### - TODO: Please change the code for MADDPG to allow for some kind of test percentage of data. Try to run over 1 month. output this - ###

def bayesian_train(args, data, train_keys):
    print('Begin Latent Training')

    ### - Set Debug - ###
    if args.debug:
        epochs = 500

    print('Num Stocks: %d' % len(train_keys))
    ### - Load AutoEncoder - ###
    ae_path = general_params['ae_path'] + general_params['ae_pt_epoch'] + '.pth'
    print('loading Auto Encoder from path: %s' % ae_path)
    shape = data[train_keys[0]].shape
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
        'key': train_keys,
        'rew_fn': args.reward,
        'ae': vqae,
    }

    bots = MADDPG(latent, len(train_keys) * latent, train_keys, 3, env_args, args.verbose, latent=True, latent_optimizer=ae_optimizer) #init bots
    mem = MultiAgentReplayBuffer(100000, len(train_keys) * latent, latent, 3, len(train_keys), 1)

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
            bots.get_renders(i, train_keys)
            
    ### - Model Save - ###
    bots.save_checkpoint()
    actionwriter.close()
    ### - Evaluation - ###
    raise NotImplementedError