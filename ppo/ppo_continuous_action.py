"""
Based on https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
"""

import numpy
import torch

import gymnasium as gym

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, activation=torch.nn.ReLU):
        super(MLP, self).__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.mlp(x)


class Agent(torch.nn.Module):
    def __init__(self, envs: gym.Env, optimzer:torch.optim.Optimizer=None, clip_coef=0.2, value_coef=0.5, entropy_coef=0.01):
        super(Agent, self).__init__()

        self.optimizer = optimzer
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        input_size = numpy.array(envs.single_observation_space.shape).prod()
        action_size = numpy.array(envs.single_action_space.shape).prod()

        self.actor = MLP(input_size, action_size, 64, activation=torch.nn.Tanh)
        self.critic = MLP(input_size, 1, 64, activation=torch.nn.Tanh)

        self.actor_std = torch.nn.Parameter(torch.zeros(1, action_size))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_mean = action_mean.reshape(-1, action_mean.shape[-1])
        action_log_std = self.actor_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        if action.shape[0] == 1:
            action = action.squeeze(0)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def update(self, states, actions, log_probs, advantages, returns, update_epochs=10, mini_batch_size=64):
        dataset = torch.utils.data.TensorDataset(states, actions, log_probs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

        for _epoch in range(update_epochs):
            for batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns in dataloader:
                _action, new_log_probs, entropy, new_value = self.get_action_and_value(batch_states, batch_actions)
                log_ratio = new_log_probs - batch_log_probs
                ratio = torch.exp(log_ratio)

                # Eq. 7. Section 5 of PPO paper
                # The sign of the advantages, value coefficient and entropy coefficient
                # is reversed compared to the paper because of the way the optimizer works
                # (also the max function is orginialy min in the paper)
                policy_gradient_loss1 = -batch_advantages * ratio
                policy_gradient_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_gradient_loss = torch.max(policy_gradient_loss1, policy_gradient_loss2).mean()

                value_loss = 0.5 * ((batch_returns - new_value) ** 2).mean()

                # Eq. 9. Section 5 of PPO paper
                loss = policy_gradient_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
