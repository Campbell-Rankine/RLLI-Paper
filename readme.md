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
Using the VGG-16 model we change the final output classification to a vector of size N. Model will attempt to learn it's own indicators by attempting to encode the 'health' of a stock using some window size W.

## Reward Function Names and Definitions
---

#### Base
---
Name: Base
Definitions: Rew_{base} = ...

Notes: This reward function will be temporary until a correct maximization function will be coded. Until then, this incorrect function will be a place holder.

## MADDPG:
---

Currently the action space is limited to a discrete setting within agent environments. This is to simplify the environment development process for the time being. 

#### Future Development:
---

Continuous action space, similar definition of buy sell Enum objects. Environment constraints class allowing for things like clipping the number of stocks each agent can trade at one timestep, limiting the variance of the portfolio etc.
