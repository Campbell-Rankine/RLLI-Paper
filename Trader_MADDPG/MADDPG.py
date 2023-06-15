import torch as T
import torch.nn.functional as F
from Trader_MADDPG.network import Agent
from Env import TradingEnv
from config import *
import matplotlib.pyplot as plt

class MADDPG:
    def __init__(self, actor_dims, critic_dims, stock_keys, n_actions, env_args: dict, env_args_t: dict, verbose,
                 scenario='s&p500',  timestep_0=30, alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, cp_='/Users/bigc/RLLI-Paper/checkpoint/', latent=False, latent_optimizer=None):
        """
        Actual Class containing all agents. See network file for information on params
        """
        ### - Copy Attributes - ###
        self.stock_keys = stock_keys
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
        self.latent=latent
        
        ### - Objects - ###
        if not self.latent:
            self.encoder_loss = 0
        self.agents = []
        self.obs_p = []
        self.n_agents = len(stock_keys)
        self.latent_optimizer = latent_optimizer
        
        ### - init the agents list - ###
        for i in range(len(stock_keys)):
            env_args['key'] = stock_keys[i]
            env_args_t['key'] = stock_keys[i] 
            self.agents.append(Agent(TradingEnv(**env_args), TradingEnv(**env_args_t), self.actor_dims, self.critic_dims, self.n_actions, self.n_agents, stock_keys[i], 
                                    self.verbose, alpha=self.alpha, beta=self.beta, fc1=self.fc1, fc2=self.fc2, gamma=self.gamma, 
                                    tau=self.tau))
            self.obs_p.append(np.zeros(self.actor_dims))

    def obs_format(self, obs):
        state = obs[0][:,0,0]
        for obs_ in obs[1:]:
            state = np.hstack([state, obs_[:,0,0]])
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
        """
        This functions steps through one training iteration of the main loop and returns the relevant
        info needed inside the main file. Done in here since overall the project structure
        suggests using environments inside their respective classes instead of the main loop.
        """
        observations = []; actions=[]; step_rewards=[]; _dones=[]; infos=[]
        skips = []; probs = []
        for i, agent in enumerate(self.agents):
            if i in skips: #skip if this stock is done, continue training on other steps
                continue
            observation, action, step_reward, _done, info, prob = agent.next_step() #call environment run
            observations.append(observation) #build all observations
            actions.append(action)
            step_rewards.append(step_reward)
            _dones.append(_done)
            infos.append(info)
            probs.append(prob)
        
        ### - Final Processing - ###
        if not any(_dones):
            state= self.obs_format(observations)
            state_p = self.obs_format(self.obs_p)

            ### - Store - ###
            if self.latent:
                raw = [x[:,0,0] for x in self.obs_p]
                new = [x[:,0,0] for x in observations]
                memory.store_transition(raw, state_p, actions, step_rewards, new, state, _dones)
            else:
                memory.store_transition(self.obs_p, state_p, actions, step_rewards, observations, state, _dones)

            if total_steps % 100 == 0:
                self.learn(memory)

            self.obs_p = observations

            score += sum(step_rewards)

        total_steps += 1
        episode_steps += 1
        
        return score, total_steps, episode_steps, infos, _dones, probs, actions
    
    def step_eval(self, memory, total_steps, episode_steps, score):
        """
        This functions steps through one training iteration of the main loop and returns the relevant
        info needed inside the main file. Done in here since overall the project structure
        suggests using environments inside their respective classes instead of the main loop.
        """
        observations = []; actions=[]; step_rewards=[]; _dones=[]; infos=[]
        skips = []; probs = []
        for i, agent in enumerate(self.agents):
            if i in skips: #skip if this stock is done, continue training on other steps
                continue
            observation, action, step_reward, _done, info, prob = agent.next_step(test=True) #call environment run
            observations.append(observation) #build all observations
            actions.append(action)
            step_rewards.append(step_reward)
            _dones.append(_done)
            infos.append(info)
            probs.append(prob)
            if _done:
                break

        if not any(_dones):
            score += sum(step_rewards)

        total_steps += 1
        episode_steps += 1
        
        return score, total_steps, episode_steps, infos, _dones, probs, actions

    def choose_action(self, raw_obs):
        actions = []
        probs = []
        for agent_idx, agent in enumerate(self.agents):
            action, prob = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
            probs.append(prob)
        return actions, probs

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
        encoder_loss = []
        critic_losses = []
        actor_losses = []
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*(critic_value_.detach().clone())
            agent.critic_loss = F.mse_loss(target, critic_value)

            agent.critic.optimizer.zero_grad()
            agent.actor.optimizer.zero_grad()

            agent.critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.actor_loss = agent.critic.forward(states, mu).flatten().mean()
            agent.actor_loss.backward(retain_graph=True)
            
            agent.actor.optimizer.step()

            agent.update_network_parameters()
            with T.no_grad():
                actor_losses.append(agent.actor_loss.item())
                critic_losses.append(agent.critic_loss.item())
        if self.latent and not self.latent_optimizer is None:
            self.latent_optimizer.zero_grad()
            encoder_loss = T.tensor(critic_losses, dtype=T.float, requires_grad=True).mean() / np.max(critic_losses)
            encoder_loss.backward()
            self.latent_optimizer.step()
            with T.no_grad():
                self.critic_loss = np.mean(critic_losses) / np.max(critic_losses)
                self.actor_loss = np.mean(actor_losses)
                self.encoder_loss = encoder_loss.item()
        else:
            with T.no_grad():
                self.critic_loss = np.mean(critic_losses) / np.max(critic_losses)
                self.actor_loss = np.mean(actor_losses)

    def update_environments(self):
        raise NotImplementedError

    def get_renders(self, iteration, tickers):
        print('Rendering Decision History')
        for x in self.agents:
            if x.stock in tickers:
                fpath = general_params['render_save'] + x.name + '_' + str(iteration) + '.png'
                x.env.render_all()
                x.env.save_rendering(fpath)
            plt.clf()

    def _get_collab_reward(self):
        raise NotImplementedError

    def reset_environments(self, epoch, mem, test=False):
        """if epoch % 20 == 0:
            mem.reset()"""
        self.obs_p = []
        for x in self.agents:
            if not test:
                x.reset()
            else:
                x.reset_test()
            self.obs_p.append(x.obs)