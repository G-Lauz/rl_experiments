import abc

import gymnasium
import numpy
import torch

from .mlp import MLP

class Agent(abc.ABC):

    action_space: gymnasium.Space
    observation_space: gymnasium.Space

    def __init__(self, env: gymnasium.Env, clip_coef=0.2, value_coef=0.5, entropy_coef=0.01):
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.observation_space_size = numpy.array(env.single_observation_space.shape).prod()
        self.action_space_size = numpy.array(env.single_action_space.shape).prod()

        self.actor = self.create_actor()
        self.critic = self.create_critic()
        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        probs = self._get_action_distribution(x)

        if action is None:
            action = probs.sample()

        if action.shape[0] == 1:
            action = action.squeeze(0)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    @abc.abstractmethod
    def create_actor(self):
        pass

    @abc.abstractmethod
    def create_critic(self):
        pass

    @abc.abstractmethod
    def _get_action_distribution(self, x):
        pass


class DiscreteAgent(Agent):
    def create_actor(self):
        return MLP(self.observation_space_size, self.action_space_size, 64, activation=torch.nn.Tanh)

    def create_critic(self):
        return MLP(self.observation_space_size, 1, 64, activation=torch.nn.Tanh)
    
    def _get_action_distribution(self, x):
        action_logits = self.actor(x)
        return torch.distributions.Categorical(logits=action_logits)


class ContinuousAgent(Agent):
    def __init__(self, env: gymnasium.Env, clip_coef=0.2, value_coef=0.5, entropy_coef=0.01):
        super(ContinuousAgent, self).__init__(env, clip_coef, value_coef, entropy_coef)

        self.actor_std = torch.nn.Parameter(torch.zeros(1, self.action_space_size))

    def create_actor(self):
        return MLP(self.observation_space_size, self.action_space_size, 64, activation=torch.nn.Tanh)

    def create_critic(self):
        return MLP(self.observation_space_size, 1, 64, activation=torch.nn.Tanh)
    
    def _get_action_distribution(self, x):
        action_mean = self.actor(x).reshape(-1, self.action_space_size)

        action_log_std = self.actor_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return torch.distributions.Normal(action_mean, action_std)
