import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Trader_MADDPG.utils import *
from Env import *
from config import *

### - Note we will have 1 agent per stock. Stock environments require a continuous output - ###

class Actor(nn.Module):
    def __init__(self, alpha, in_size, fc1, fc2, n_actions, name, dir='/home/campbell/Desktop/Python-Projects/RLLI-Paper/checkpoint'):
        """
        Actor Class:

        Args:
            alpha:          (float) - Optimization parameter for actor network
            in_size:        (list - (int)) - Input Dimensions corresponding w/ env observation dims
            fc1:            (int) - Dimension for first fully connected layer
            fc2:            (int) - Dimension for second fully connected layer
            n_actions:      (int) - output dim for action vector
            name:           (str) - 'member' name (usually should correspond to stock tracked)
        """
        super(Actor, self).__init__()

        self.cp_ = os.path.join(dir, name)

        ### - Attribute copies - ###
        self.in_size = in_size
        self.n_actions = n_actions
        self.alpha = alpha

        ### - Network Inits - ###
        self.fc1 = nn.Linear(in_size, fc1)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight, -f1, f1) #weight init for stability
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.ln1 = nn.LayerNorm(fc1)

        self.fc2 = nn.Linear(fc1, fc2)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight, -f2, f2) #weight init for stability
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.ln2 = nn.LayerNorm(fc2) #layer norm

        self.pi = nn.Linear(fc2, n_actions)
        f3 = 0.003
        nn.init.uniform_(self.pi.weight, -f3, f3) #weight init for stability
        nn.init.uniform_(self.pi.bias.data, -f3, f3)

        ### - Other Training Objects - ###
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.activation = T.softmax
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        pi = self.activation(self.pi(x), dim=1)
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.cp_)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.cp_))

class Critic(nn.Module):
    def __init__(self, beta, in_size, fc1, fc2, n_actions, n_agents, name, dir='/home/campbell/Desktop/Python-Projects/RLLI-Paper/checkpoint'):
        """
        Critic Class:

        Args:
            beta:          (float) - Optimization parameter for critic network
            in_size:        (list - (int)) - Input Dimensions corresponding w/ env observation dims
            fc1:            (int) - Dimension for first fully connected layer
            fc2:            (int) - Dimension for second fully connected layer
            n_actions:      (int) - output dim for action vector
            name:           (str) - 'member' name (usually should correspond to stock tracked)
        """
        super().__init__()

        self.cp_ = os.path.join(dir, name)

        ### - Attribute copies - ###
        self.in_size = in_size
        self.n_actions = n_actions
        self.beta = beta
        self.n_agents = n_agents

        ### - Network Inits - ###
        self.fc1 = nn.Linear(in_size, fc1)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight, -f1, f1) #weight init for stability
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.ln1 = nn.LayerNorm(fc1) #layer norm

        self.fc2 = nn.Linear(fc1, fc2)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight, -f2, f2) #weight init for stability
        T.nn.init.uniform_(self.fc1.bias.data, -f2, f2)

        self.ln2 = nn.LayerNorm(fc2) #layer norm

        self.action_value = nn.Linear(self.n_actions * self.n_agents, fc2) #Action Value

        f3 = 0.003
        self.q = nn.Linear(fc2, 1) # Actor Gradient Approximation
        nn.init.uniform_(self.q.weight, -f3, f3) #weight init for stability
        nn.init.uniform_(self.fc1.bias.data, -f3, f3)

        ### - Other Training Objects - ###
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.activation = T.softmax
 
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.ln1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.ln2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(state_value + action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.cp_)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.cp_))

### - Individual Agents - ###
class Agent(nn.Module):
    def __init__(self, env: TradingEnv, actor_dims, critic_dims, n_actions, n_agents, stock, verbose, dir='/home/campbell/Desktop/Python-Projects/RLLI-Paper/checkpoint',
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01):
        """
        Agent Class: Comprised of both actor and critic networks, the agent will interact with enviromnent and
                     will maximize the reward function.

        Credits to Phil Tabour, a lot of the code is heavily based off of the MADDPG video he did
        however there have been changes added for continuous stability.

        Args:
            env:            (OpenAI.Gym) - Gym environment for single stock action
            actor dims:     (list - (int)) - Input dim for actor init
            critic_dims:    (list - (int)) - Input dim for critic init
            n_actions:      (int) - output dim for action vector
            stock           (str) - helps name the agent (dataset index key)
            dir             (str) - checkpoint directory location
            fc1:            (int) - Dimension for first fully connected layer
            fc2:            (int) - Dimension for second fully connected layer
            alpha:          (float) - Actor optimization param
            beta:           (float) - Critic optimization param
            gamma:          (float) - Discount Factor (original paper = 0.95)
            tau:            (float) - Soft update parameter (weights the merge of prev and best iteration)
        """
        super(Agent, self).__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        ### - Copy Attributes - ###
        self.verbose = verbose
        self.gamma = gamma
        self.tau = tau
        self.n_actions = int(n_actions)
        self.name = 'agent_' + stock
        self.env = env
        self.obs = self.env.display_config(self.verbose)
        self.timestep = self.env._current_tick
        self.n_agents = n_agents
        self.actor_loss = None
        self.critic_loss = None
        self.stock = stock

        ### - Create Networks - ###
        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))

        self.actor = Actor(alpha, actor_dims, fc1, fc2, self.n_actions, self.name+'_actor')
        self.critic = Critic(beta, critic_dims, fc1, fc2, self.n_actions, self.n_agents, self.name+'_critic')

        self.target_actor = Actor(alpha, actor_dims, fc1, fc2, self.n_actions, self.name+'_target_actor')
        self.target_critic = Critic(beta, critic_dims, fc1, fc2, self.n_actions, self.n_agents, self.name+'_target_critic')

    def reset(self):
        self.obs = self.env.reset()
        self.timestep = self.env._start_tick

    def next_step(self, test=False):
        probs = self.predict_proba()
        action = self.choose_action()
        observation, step_reward, _done, info = self.env.step(action)
        self.obs = observation
        return observation, action, step_reward, _done, info, probs

    def choose_action(self):
        state = T.tensor(self.obs, dtype=T.float).T#.to(self.actor.device)
        actions = self.actor.forward(state.to(self.device))
        actions = actions.detach().cpu().numpy() + self.noise()
        actions = actions[0][0]
        max_a = np.argmax(actions)
        if max_a == 0:
            return 1*(general_params['max_action'] * actions[0])
        if max_a == 1:
            return -1*(general_params['max_action'] * actions[0])
        return 0
    
    def predict_proba(self):
        """
        Predict probabilities wrapper for numpy/sklearn library integration (Not used as part of my personal implementation but included for convenience)
        """
        state = T.tensor(self.obs, dtype=T.float).T#.to(self.actor.device)
        actions = self.actor.forward(state.to(self.device))
        actions = actions.detach().cpu().numpy()
        actions = actions[0][0]
        return actions
    
    def predict(self):
        """
        Predict wrapper for the SHAP class and other numpy integrations
        """
        return self.choose_action()

    def update_network_parameters(self, tau=None):
        """
        Function takes some kind of weight (Tau) and will merge weights and parameters from the normal -> target networks
        """
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.parameters() #Access all target/normal params for soft update
        actor_params = self.actor.parameters()
        for t_param, param in zip(target_actor_params, actor_params):
            t_param.data.copy_(tau*param + (1-tau)*t_param) #Copy the soft update to the target model

        target_critic_params = self.target_critic.parameters() #Access all target/normal params for soft update
        critic_params = self.critic.parameters() #Copy the soft update to the target model

        for t_param, param in zip(target_critic_params, critic_params):
            t_param.data.copy_(tau*param + (1-tau)*t_param)

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