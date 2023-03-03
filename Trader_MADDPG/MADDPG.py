import torch as T
import torch.nn.functional as F
from Trader_MADDPG.network import Agent
from Env import *
from config import *

class MADDPG:
    def __init__(self, actor_dims, critic_dims, stock_keys, n_actions, env_args: dict, verbose,
                 scenario='s&p500',  timestep_0=30, alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, cp_='/Users/bigc/RLLI-Paper/checkpoint/'):
        """
        Actual Class containing all agents. See network file for information on params
        """
        ### - Copy Attributes - ###
        self.verbose = verbose
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.fc1 = fc1
        self.fc2 = fc2
        self.cp_ = cp_
        self.timestep_0 = timestep_0
        self.current_t = timestep_0
        
        ### - Objects - ###
        self.agents = []
        self.obs_p = []
        self.n_agents = len(stock_keys)
        
        ### - init the agents list - ###
        for i in range(len(stock_keys)):
            env_args['key'] = stock_keys[i]
            self.agents.append(Agent(TradingEnv(**env_args), self.actor_dims, self.critic_dims, self.n_actions, self.n_agents, stock_keys[i], 
                                    self.verbose, alpha=self.alpha, beta=self.beta, fc1=self.fc1, fc2=self.fc2, gamma=self.gamma, 
                                    tau=self.tau))
            self.obs_p.append(np.zeros(self.actor_dims))

    def obs_format(self, obs):
        state = obs[0]
        for obs_ in obs[1:]:
            state = np.hstack([state, obs_])
        return state

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def step(self, memory, total_steps, episode_steps, score):
        #TODO: This is the main function that needs testing
        """
        This functions steps through one training iteration of the main loop and returns the relevant
        info needed inside the main file. Done in here since overall the project structure
        suggests using environments inside their respective classes instead of the main loop.
        """
        observations = []; actions=[]; step_rewards=[]; _dones=[]; infos=[]
        skips = []
        for i, agent in enumerate(self.agents):
            if i in skips: #skip if this stock is done, continue training on other steps
                continue
            observation, action, step_reward, _done, info = agent.next_step() #call environment run
            observations.append(observation) #build all observations
            actions.append(action)
            step_rewards.append(step_reward)
            _dones.append(_done)
            infos.append(info)
        
        ### - Final Processing - ###
        state= self.obs_format(observations)
        state_p = self.obs_format(self.obs_p)

        ### - Store - ###
        memory.store_transition(self.obs_p, state_p, actions, step_rewards, observations, state, _dones)

        if total_steps % 100 == 0:
            self.learn(memory)

        self.obs_p = observations

        score += sum(step_rewards)

        total_steps += 1
        episode_steps += 1
        
        return score, total_steps, episode_steps, infos, _dones

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts.detach().clone() for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*(critic_value_.detach().clone())
            agent.critic_loss = F.mse_loss(target, critic_value)

            agent.critic.optimizer.zero_grad()
            agent.actor.optimizer.zero_grad()

            agent.critic_loss.backward(retain_graph=True)
            agent.actor_loss = -agent.critic.forward(states, mu).flatten().mean()
            agent.actor_loss.backward(retain_graph=True)
            
            agent.critic.optimizer.step()
            agent.actor.optimizer.step()

            agent.update_network_parameters()

    def update_environments(self):
        raise NotImplementedError

    def get_renders(self, iteration):
        print('Rendering Decision History')
        for x in self.agents:
            fpath = general_params['render_save'] + x.name + '_' + str(iteration) + '.png'
            x.env.render_all()
            x.env.save_rendering(fpath)
            plt.clf()

    def _get_collab_reward(self):
        raise NotImplementedError

    def reset_environments(self):
        self.obs_p = []
        for x in self.agents:
            x.reset()
            self.obs_p.append(x.obs)