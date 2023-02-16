import torch as T
import torch.nn as nn
import torch.nn.functional as F

from Trader_single.utils import *

import os
import ray

### - DEFINE ACTOR CRITIC NETWORKS - ###

class Actor(nn.Module):
    def __init__(self, h1, h2, obs_size, action_size, batch_size, tau, betas, lr, cp, name, device):
        super(Actor, self).__init__()
        
        ### - Attributes - ###
        self.obs_size = obs_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.cp = cp
        self.cp_save = os.path.join(cp, name)
        self.name = name

        ### - Network Modules - ###
        self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(self.obs_size, h1)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight, -f1, f1)

        self.ln1 = nn.LayerNorm(h1)

        self.fc2 = nn.Linear(h1, h2)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight, -f2, f2)

        self.ln2 = nn.LayerNorm(h2)

        self.mu = nn.Linear(h2, self.action_size)
        f3 = 0.003
        nn.init.uniform_(self.mu.weight, -f3, f3)

        print('Weight init sample range for %s:' % self.name)
        print('-------------------------------------------------')
        print('L1: U(%.3f, %.3f), L2: U(%.3f, %.3f), Mu: U(%.3f, %.3f)' % (-f1, f1, -f2, f2, -f3, f3))
        print()

        self.optim = T.optim.Adam(self.parameters(), lr=lr, betas=betas)
        self.device = device

    def forward(self, state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x)
        x = self.mu(x)
        return x.mean(0)
    
    # EVENTUALLY YOUR ENCODER SHOULD TAKE THE INPUT FOR STATE, TRY TO DESCRIBE MARKET HEALTH
    #TODO: Make the autoencoder follow the same structure 

    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        try:
            T.save(self.state_dict(), self.cp_save)
        except RuntimeError:
            os.mkdir(self.cp)
            T.save(self.state_dict(), self.cp_save + '.pth')

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        try:
            self.load_state_dict(T.load(self.cp_save + '.pth'))
        except Exception:
            print('path does not exist!')

class Critic(nn.Module):
    def __init__(self, h1, h2, obs_size, action_size, batch_size, w_decay, tau, betas, lr, cp, name, device):
        super(Critic, self).__init__()
        
        ### - Attributes - ###
        self.obs_size = obs_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.cp = cp
        self.cp_save = os.path.join(cp, name)
        self.w_decay = w_decay
        self.name = name

        ### - Network Modules - ###
        self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(self.obs_size, h1)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight, -f1, f1)

        self.ln1 = nn.LayerNorm(h1)

        self.fc2 = nn.Linear(h1, h2)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight, -f2, f2)

        self.ln2 = nn.LayerNorm(h2)

        self.fc3 = nn.Linear(h2, self.action_size)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight, -f2, f2)

        self.ln3 = nn.LayerNorm(self.action_size)

        self.q = nn.Linear(self.action_size, 1)
        f3 = 0.003
        nn.init.uniform_(self.q.weight, -f3, f3)

        print('Weight init sample range for %s:' % self.name)
        print('-------------------------------------------------')
        print('L1: U(%.3f, %.3f), L2: U(%.3f, %.3f), Q: U(%.3f, %.3f)' % (-f1, f1, -f2, f2, -f3, f3))
        print()

        self.av = nn.Linear(self.action_size, self.action_size)

        self.optim = T.optim.Adam(self.parameters(), lr=lr, betas=betas, weight_decay=self.w_decay)
        self.device = device

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.ln1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.ln2(state_value)
        state_value = self.fc3(state_value)
        state_value = self.ln3(state_value)

        action_value = self.activation(self.av(action))
        state_action_value = self.activation(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        try:
            T.save(self.state_dict(), self.cp_save)
        except RuntimeError:
            os.mkdir(self.cp)
            T.save(self.state_dict(), self.cp_save + '.pth')

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        try:
            self.load_state_dict(T.load(self.cp_save + '.pth'))
        except Exception:
            print('path does not exist!')

### - DEFINE AGENT - ###
from Trader_single.buffer import *
class Agent(nn.Module):
    def __init__(self, alpha, beta, lr, dims, tau, cp, name, gamma=0.99, num_actions=138, obs_size=6,  max_size=1000000, h1=400,
                 h2=500, batch_size=30, w_decay=0.1):
        super(Agent, self).__init__()
        ### - ATTRIBUTES - ###
        self.lr = lr
        self.holdings = 50
        self.max_num_stocks = 50
        self.alpha = alpha
        self.beta = beta
        self.dims = dims
        self.tau = tau
        self.gamma = gamma
        self.obs_size = 834
        self.num_actions = num_actions
        self.max_size = max_size
        self.h1 = h1
        self.h2 = h2
        self.batch_size = batch_size
        self.cp = cp
        self.name = name
        self.weight_decay = w_decay
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        ### - Network Inits - ###
        self.memory = ReplayBuffer(self.max_size, (30,834), num_actions)

        self.actor = Actor(self.h1, self.h2, self.obs_size, self.num_actions, self.batch_size, self.tau, self.alpha, 
                           self.lr, self.cp, self.name + '_actor', self.device)
        
        self.critic = Critic(self.h1, self.h2, self.obs_size, self.num_actions, self.batch_size, self.weight_decay, self.tau, self.beta, 
                           self.lr, self.cp, self.name + '_critic', self.device)
        
        self.target_actor = Actor(self.h1, self.h2, self.obs_size, self.num_actions, self.batch_size, self.tau, self.alpha, 
                           self.lr, self.cp, self.name + '_target_actor', self.device)
        
        self.target_critic = Critic(self.h1, self.h2, self.obs_size, self.num_actions, self.batch_size, self.weight_decay, self.tau, self.beta, 
                           self.lr, self.cp, self.name + '_target_critic', self.device)
        
        self.noise = OrnsteinUhlenbeckActionNoise(self.num_actions)

        self.storage = plot_mem(50)

        self.update_(tau=1.)

    ### - Training update functions - ###
    def update_(self, tau=None):
        if tau is None:
            tau = self.tau
        
        ### - Network tau adjustment - ###
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float)
        mu = self.actor.forward(observation)
        #print(mu.shape)
        sample = self.noise.sample()
        self.sample_mag = T.norm(T.tensor(sample))
        #print(sample.shape)
        self.storage.add(sample)
        mu_prime = mu + T.tensor(sample,
                                 dtype=T.float)
        mu_prime = mu_prime.clip(0., self.max_num_stocks)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        #if np.mean(reward) > 1e-4:
        self.memory.add(state, action, reward, new_state, done)

    def learn(self):
        #TODO: FIX LEARN FUNCTION
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample(self.batch_size)
        reward = T.tensor(reward, dtype=T.float)#.to(self.critic.device)
        done = T.tensor(done)#.to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float)#.to(self.critic.device)
        action = T.tensor(action, dtype=T.float)#.to(self.critic.device)
        state = T.tensor(state, dtype=T.float)#.to(self.critic.device)

        ### - CPU parallel - ###
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)
        
        ### Secondary CPU parallel loop - ###
        target = []
        for j in range(self.batch_size):
            toapp = reward[j] + self.gamma*critic_value_[j]*done[j]
            target.append(toapp.mean().item())
        target = T.tensor(target)#.to(self.critic.device)
        #print(target)
        target = target.view(self.batch_size, 1)
        
        ### - Optimize Section: Can also be run in parallel - ###
        self.critic.train()
        self.critic.optim.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()

        nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=10.0)

        self.critic.optim.step()

        self.critic.eval()
        self.actor.optim.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()

        nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=10.0)

        self.actor.optim.step()
        #self.gamma *= self.gamma
        self.update_()


    ### - Helpers - ###
    def update_holdings(self, new_vec):
        self.holdings = new_vec

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()


#FTranspose input