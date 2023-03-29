# Reinforcement Learning - Learned Indicators
---

This repository aims to train a 'portfolio management' algorithm, using learned indicators for smart profit/risk balancing. The Environment code was originally taken from the AnyTrading repository linked below. Stocks will be iterated through over time, and as long as the dataset is in a dictionary of DataFrames format, you should be able to use any tickers you'd like.

Still very much a work in progress.

## Model Structure
---
Continuous Space Deep Deterministic Policy Gradients is comprised of an actor, critic, target actor, and target critic network. The configuration of actor == target actor, and similarly the configuration of critic == target critic. The solution to the exploitation vs exploration problem is given by a OU noise sampler that samples temporally local points from a normal distribution. The model is optimized by maximizing the continuous Bellman Equation.

#### Actor
---
Using input observation x at time t, we try to output the solution to the maximum likelihood estimation of the reward function. Consists of a continuous action/reward space.

#### Critic
---
Self supervised optimizer. Attempts to estimate the gradient of the reward function wrt. the weights of the actor model. Critic is optimized by the bellman equation and outputs the loss value of the actor. 

## Indicator Frameworks
---
For experiment purposes we use a 'sliding scale' of learned indicators. The sliding scale describes the involvement of learned parameters in the indicator variables.

#### Standard Indicators
---
Using the quantstats package we take the base 138 indicators provided in the data utils package. Will be a baseline comparison for future model implementations

#### Learned indicator weighting
---
Using the same quantstats package we weight all 138 indicators based on their importance throughout timesteps. The technique is best described as a learned weighted moving average across window size W.

#### Encoded Indicators
---
Using the VQVAE model we change the final output classification to a vector of size N. Model will attempt to learn it's own indicators by attempting to encode the 'health' of a stock using some window size W.

The network architecture will be included below:

TODO: INSERT IMAGE

The loss function for our windowed auto encoder is the standard combination of reconstruction error and embedding loss. However it is important to note that the loss function has been changed as we are simply pre training an auto encoder so the actual embedding/reconstruction loss is relatively arbitrary at this point. To allow for task specific training we have provided input parameters inside the config file that turn the loss function into a weighted combination of the embedding and reconstruction loss. 0.8 for reconstruction and 0.2 for embedding loss are the default weights however training the model at 0.9 and 0.1 is recommended as it reduces the magnitude of the loss function and yields slightly better training performance. Another good strategy for training is to simply pick one or the other. In the case that one of the losses has a weight of 0,  the calculations for this loss will be skipped to minimize unnecessary computations.

## Reward Function Names and Definitions
---

#### Base
---
Name: Base 

Definitions: Normalized Net Worth at the next timestep.

Basis: Following Markov Decision Process fundamentals and definitions we reward the model for the normalized net worth at timestep T+1

---

Name: Base Profit

Definitions: Mean discounted net worth across all future timesteps

Basis: Similar logic of following the MDP definitions however now we add long/short term discounting

## MADDPG:
---

Currently the action space is limited to a discrete setting within agent environments. This is to simplify the environment development process for the time being. 

#### Future Development:
---

Continuous action space, similar definition of buy sell Enum objects. Environment constraints class allowing for things like clipping the number of stocks each agent can trade at one timestep, limiting the variance of the portfolio etc.
