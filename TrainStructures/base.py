from config import *
from config import *
import tqdm._tqdm
import argparse

from data import *

from Trader_MADDPG.MADDPG import *
from Trader_MADDPG.buffer import *
from AE.AE import *
from torch import optim

from torch.utils.tensorboard import SummaryWriter

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