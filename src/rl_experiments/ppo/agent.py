"""
Based on https://github.com/vwxyzjn/ppo-implementation-details.git
"""
import abc

import gymnasium
import numpy
import torch

from rl_experiments.config import Config
from rl_experiments.model import MLP


class Agent(abc.ABC, torch.nn.Module):
    action_space: gymnasium.Space
    observation_space: gymnasium.Space

    def __init__(self, observation_space: gymnasium.Space, action_space: gymnasium.Space, config: Config):
        super(Agent, self).__init__()

        self.clip_coef = config.agent.clip_coef
        self.value_coef = config.agent.value_coef
        self.entropy_coef = config.agent.entropy_coef
        self.config = config

        self.observation_space = observation_space
        self.action_space = action_space

        self.observation_space_size = int(numpy.array(observation_space.shape).prod())
        self.action_space_size = int(numpy.array(action_space.shape).prod())

        self.actor = None
        self.critic = None

        self.optimizer = None

    def initilaize(self):
        self.actor = self.create_actor()
        self.critic = self.create_critic()

        self.optimizer = self.configure_optimizers()

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        probs = self._get_action_distribution(x)

        if action is None:
            action = probs.sample()

        if isinstance(self.action_space, gymnasium.spaces.Box):
            if action.shape[0] == 1:
                action = action.squeeze(0)

        if isinstance(self.action_space, gymnasium.spaces.Box):
            log_prob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            log_prob = probs.log_prob(action)
            entropy = probs.entropy()
        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported")

        return action, log_prob, entropy, self.critic(x)

    def predict(self, x, deterministic=False):
        """Based on stable_baselines3: https://github.com/DLR-RM/stable-baselines3/blob/4af4a32d1b5acb06d585ef7bb0a00c83810fe5c3/stable_baselines3/common/policies.py#L331"""
        self.train(False)

        with torch.no_grad():
            # observation to tensor
            device = torch.device("cuda" if self.config.train.cuda else "cpu") # TODO: Patch, fix this
            x = torch.as_tensor(x, device=device, dtype=torch.float32)

            probs = self._get_action_distribution(x)

            if deterministic:
                if isinstance(probs, torch.distributions.Categorical):
                    actions = probs.argmax(dim=-1, keepdim=True)
                elif isinstance(probs, torch.distributions.Normal):
                    actions = probs.mean
                else:
                    raise NotImplementedError(f"Action distribution {probs} not supported")
            else:
                actions = probs.sample()

            actions = actions.cpu().numpy().reshape(-1, *self.action_space.shape)

            if isinstance(self.action_space, gymnasium.spaces.Box):
                actions = numpy.clip(actions, self.action_space.low, self.action_space.high)

            # Check if the observation is vectorized
            if not self.is_vectorized(x):
                assert isinstance(actions, numpy.ndarray)
                actions = actions.squeeze(axis=0)

        return actions, None # None for compatibility with stable_baseline3 predict method

    def is_vectorized(self, x):
        """Based on stable_baselines3: https://github.com/DLR-RM/stable-baselines3/blob/4af4a32d1b5acb06d585ef7bb0a00c83810fe5c3/stable_baselines3/common/utils.py#L381"""
        if isinstance(self.observation_space, gymnasium.spaces.Box):
            if x.shape == self.observation_space.shape:
                return False
            elif x.shape[1:] == self.observation_space.shape:
                return True
            else:
                raise ValueError(f"Invalid shape for box observation {x.shape}")
        elif isinstance(self.observation_space, gymnasium.spaces.Discrete):
            if isinstance(x, int) or x.shape == ():
                return False
            elif len(x.shape) == 1:
                return True
            else:
                raise ValueError(f"Invalid shape for discrete observation {x.shape}")
        else:
            raise NotImplementedError(f"Unsupported observation space {self.observation_space}")

    def save(self, path):
        torch.save(self.state_dict(), path)

    @abc.abstractmethod
    def configure_optimizers(self):
        pass

    @abc.abstractmethod
    def create_actor(self):
        pass

    @abc.abstractmethod
    def create_critic(self):
        pass

    @abc.abstractmethod
    def _get_action_distribution(self, x) -> torch.distributions.Distribution:
        pass


class DiscreteAgent(Agent):
    def __init__(self, observation_space: gymnasium.Space, action_space: gymnasium.Space, config: Config):
        super(DiscreteAgent, self).__init__(observation_space, action_space, config)

        self.action_space_size = action_space.n # TODO: Patch, fix this

        self.hidden_size = config.agent.hidden_size
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.train.learning_rate, eps=1e-5)

    def create_actor(self):
        return MLP(self.observation_space_size, self.action_space_size, self.hidden_size, activation=torch.nn.Tanh)

    def create_critic(self):
        return MLP(self.observation_space_size, 1, self.hidden_size, activation=torch.nn.Tanh)

    def _get_action_distribution(self, x) -> torch.distributions.Distribution:
        action_logits = self.actor(x)
        return torch.distributions.Categorical(logits=action_logits)


class ContinuousAgent(Agent):
    def __init__(self, observation_space: gymnasium.Space, action_space: gymnasium.Space, config: Config):
        super(ContinuousAgent, self).__init__(observation_space, action_space, config)

        self.hidden_size = config.agent.hidden_size

        self.actor_std = torch.nn.Parameter(torch.zeros(1, self.action_space_size))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.train.learning_rate, eps=1e-5)

    def create_actor(self):
        return MLP(self.observation_space_size, self.action_space_size, self.hidden_size, activation=torch.nn.Tanh)

    def create_critic(self):
        return MLP(self.observation_space_size, 1, self.hidden_size, activation=torch.nn.Tanh)

    def _get_action_distribution(self, x) -> torch.distributions.Distribution:
        action_mean = self.actor(x).reshape(-1, self.action_space_size)

        action_log_std = self.actor_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return torch.distributions.Normal(action_mean, action_std)
