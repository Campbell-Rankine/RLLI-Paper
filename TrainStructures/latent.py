from config import *
from config import *
from tqdm import tqdm
import argparse
from tqdm._tqdm import *
from data import *

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


def latent_train(args, data, keys):
    print('Begin Latent Training')
    epochs = args.e

    print('Num Stocks: %d' % len(keys))
    env_params['plot_list'] = [key for i, key in enumerate(keys) if i % env_params['plot'] == 0]
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
        'ae': vqae,
    }

    bots = MADDPG(latent, len(keys) * latent, keys, 3, env_args, args.verbose, latent=True, latent_optimizer=ae_optimizer) #init bots
    mem = MultiAgentReplayBuffer(100000, len(keys) * latent, latent, 3, len(keys), 1)

    ### - START TRAIN LOOP - ###
    total_steps = 0 #Logging vars
    total_score = 0
    databar = tqdm(range(epochs+1))
    test_score = 0

    tbwriter = SummaryWriter(general_params['log dir'])

    for i in databar: #Start of main loop
         ### - Epoch Setup - ###
        bots.reset_environments(i, mem) #Basic loggers and trackers
        score = 0
        dones = [False]*bots.n_agents
        episode_steps = 0

        ### - TensorBoard Logging - ###
        buys = 0
        sells = 0
        profits = []
        while not any(dones):
            ### - Episode Steps - ###
            score, total_steps, episode_steps, infos, _dones, probs, actions = bots.step(mem, total_steps, episode_steps, score) #Run Envs
            total_score+= score
            dones = _dones
            profits.append(sum([x.env.profit for x in bots.agents]))
            buys += np.sum([x for x in actions if x > 0])
            sells += np.sum([x for x in actions if x < 0])
            if episode_steps % 50 ==0:
                databar.set_description('Epoch %d, Current Iters: %d, Episode Iters: %d, Mean Owned: %.2f. Mean Profit: %.2f, Mean Funds: %.2f, Sum Profit: %.2f, Testing Profit: %.2f' % 
                (i, total_steps, episode_steps, sum([x.env.num_owned for x in bots.agents]) / bots.n_agents, sum([x.env.profit for x in bots.agents]) / bots.n_agents, 
                sum([x.env.available_funds for x in bots.agents]) / len(bots.agents), sum([x.env.profit for x in bots.agents]), test_score )) #Logging

                
                tbwriter.add_scalar('Profit', np.sum([x.env.profit for x in bots.agents]), i)
                tbwriter.add_scalar('Spending', np.sum([x.env.available_funds for x in bots.agents]), i)
                tbwriter.add_scalars('Episode Average Action Statistics', {'Buys': buys / episode_steps, 'Sells': sells / episode_steps}, i) # Retu
        buys = 0
        sells = 0
        if i % 10 == 0 and i < 50:
            mem.reset() #reset -> Unlearn past mistakes. Don't provide bad examples provide good examples. Might be worth looking into what 
        if i % 5 == 0 and args.render:
            bots.get_renders(i, keys, profits)
    tbwriter.close()
    ### - Model Save - ###
    bots.save_checkpoint()
    ### - Evaluation - ###