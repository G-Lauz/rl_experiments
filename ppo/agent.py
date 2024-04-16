import abc

import gymnasium
import numpy
import torch

from .mlp import MLP

class Agent(abc.ABC):
    def __init__(self, env: gymnasium.Env, clip_coef=0.2, value_coef=0.5, entropy_coef=0.01):
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.observation_space_size = numpy.array(env.single_observation_space.shape).prod()
        self.action_space_size = numpy.array(env.single_action_space.shape).prod()

    @abc.abstractmethod
    def get_value(self, x):
        pass

    @abc.abstractmethod
    def get_action_and_value(self, x, action=None):
        pass


class DiscreteAgent(Agent):
    def __init__(self, env: gymnasium.Env, clip_coef=0.2, value_coef=0.5, entropy_coef=0.01):
        super(DiscreteAgent, self).__init__(env, clip_coef, value_coef, entropy_coef)

        self.actor = MLP(self.observation_space_size, self.action_space_size, 64, activation=torch.nn.Tanh)
        self.critic = MLP(self.observation_space_size, 1, 64, activation=torch.nn.Tanh)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=action_logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class ContinuousAgent(Agent):
    def __init__(self, env: gymnasium.Env, clip_coef=0.2, value_coef=0.5, entropy_coef=0.01):
        super(ContinuousAgent, self).__init__(env, clip_coef, value_coef, entropy_coef)

        self.actor = MLP(self.observation_space_size, self.action_space_size, 64, activation=torch.nn.Tanh)
        self.critic = MLP(self.observation_space_size, 1, 64, activation=torch.nn.Tanh)

        self.actor_std = torch.nn.Parameter(torch.zeros(1, self.action_space_size))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x).reshape(-1, self.action_space_size)
        action_log_std = self.actor_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        if action.shape[0] == 1:
            action = action.squeeze(0)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
