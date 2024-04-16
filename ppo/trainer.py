import abc

import gymnasium
import torch

from .agent import Agent


class Trainer(abc.ABC):
    def __init__(self, agent: Agent, env: gymnasium.Env, device: str = "cpu", num_steps: int = 1000, num_envs: int = 1,
                 batch_size: int = 64, total_timesteps: int = 25000):
        self.agent = agent
        self.env = env
        self.device = device
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.batch_size = batch_size

        self.num_updates = self.total_timesteps // self.batch_size

        # Initialize the tensors
        self.states = torch.zeros((self.num_steps, self.num_envs) + self.env.single_observation_space.shape).to(
            self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.env.single_action_space.shape).to(self.device)
        self.log_probs = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        self.rewards = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        self.values = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        self.dones = torch.zeros(self.num_steps, self.num_envs).to(self.device)

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def update_agent(self):
        pass


class DiscreteTrainer(Trainer):
    def __init__(self, agent: Agent, env: gymnasium.Env, device: str = "cpu", num_steps: int = 1000, num_envs: int = 1,
                 batch_size: int = 64):
        super(DiscreteTrainer, self).__init__(agent, env, device, num_steps, num_envs, batch_size)

    def train(self):
        for

    def update_agent(self):
        raise NotImplementedError
