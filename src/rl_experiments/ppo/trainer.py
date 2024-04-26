"""
Based on https://github.com/vwxyzjn/ppo-implementation-details.git
"""
import abc
import pathlib
from logging import Logger

import matplotlib.pyplot as plt
import numpy
import torch

from rl_experiments.config import Config
from rl_experiments.environment import EnvironmentWrapper
from rl_experiments.ppo import Agent
from rl_experiments.metrics import Metrics


class Trainer(abc.ABC):
    def __init__(self, agent: Agent, env_wrapper: EnvironmentWrapper, config: Config = None, logger: Logger = None):
        self.config = config
        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.train.cuda else "cpu")

        self.agent = agent.to(self.device)

        self.envs = env_wrapper.get_envs()

        self.buffer_dim = (self.config.train.n_steps, self.config.env.n_envs)

        self.rollout_update = self.config.train.total_timesteps // numpy.array(self.buffer_dim).prod()

        # Initialize the tensors
        self.states = torch.zeros(self.buffer_dim + env_wrapper.observation_space.shape).to(self.device)
        self.actions = torch.zeros(self.buffer_dim + env_wrapper.action_space.shape).to(self.device)
        self.log_probs = torch.zeros(self.buffer_dim).to(self.device)
        self.rewards = torch.zeros(self.buffer_dim).to(self.device)
        self.values = torch.zeros(self.buffer_dim).to(self.device)
        self.dones = torch.zeros(self.buffer_dim).to(self.device)

        self.metrics = Metrics()

    def train(self):
        self.agent.train(True)

        # rollout_update = self.config.train.rollout_update
        rollout_update = self.rollout_update
        for update in range(1, rollout_update + 1):
            self.logger.info(f"Rollout update: {update}/{rollout_update}...")

            next_state, reset_info = self.envs.reset()

            next_state = torch.Tensor(next_state).to(self.device)
            next_done = torch.zeros(self.config.env.n_envs).to(self.device)

            # Run a full episode
            for step in range(self.config.train.n_steps):
                self.states[step] = next_state
                self.dones[step] = next_done

                with torch.no_grad():
                    action, log_prob, _entropy, value = self.agent.get_action_and_value(next_state)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.log_probs[step] = log_prob

                next_state, reward, next_done, _trunc, _step_info = self.envs.step(action.cpu().numpy())
                self.rewards[step] = torch.tensor(reward).to(self.device).reshape(-1)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                next_done = torch.tensor(next_done, dtype=torch.float32).to(self.device)

            # Bootstrap the value function
            with torch.no_grad():
                last_value = self.agent.get_value(next_state).reshape(1, -1)

                if self.config.train.gae:
                    # Generalized Advantage Estimation
                    advantages = torch.zeros_like(self.rewards).to(self.device)
                    last_gae_lambda = 0.0
                    for t in reversed(range(self.config.train.n_steps)):
                        if t == self.config.train.n_steps - 1:
                            next_non_terminal = 1.0 - next_done.float()
                            next_value = last_value
                        else:
                            next_non_terminal = 1.0 - self.dones[t + 1].float()
                            next_value = self.values[t + 1]

                        delta = self.rewards[t] + self.config.train.gamma * next_non_terminal * next_value - self.values[t]
                        last_gae_lambda = delta + self.config.train.gamma * self.config.train.gae_lambda * next_non_terminal * last_gae_lambda
                        advantages[t] = last_gae_lambda

                    returns = advantages + self.values
                else:
                    returns = torch.zeros_like(self.rewards).to(self.device)
                    for t in reversed(range(self.config.train.n_steps)):
                        if t == self.config.train.n_steps - 1:
                            next_non_terminal = 1.0 - next_done.float()
                            next_return = last_value
                        else:
                            next_non_terminal = 1.0 - self.dones[t + 1].float()
                            next_return = returns[t + 1]

                        returns[t] = self.rewards[t] + self.config.train.gamma * next_non_terminal * next_return

                    advantages = returns - self.values

            # Flatten the data
            self.states = self.states.reshape((-1,) + self.envs.single_observation_space.shape)
            self.actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            self.log_probs = self.log_probs.reshape(-1)
            self.values = self.values.reshape(-1)
            advantages = advantages.reshape(-1)
            returns = returns.reshape(-1)

            self.update_agent(self.states, self.actions, self.log_probs, advantages, returns)

            # Compute the explained variance
            with torch.no_grad():
                explained_variance = 1 - (self.values - returns).var() / returns.var()

            self.metrics.explained_variance = numpy.append(self.metrics.explained_variance, explained_variance)

            # Reshape the data for next iteration
            self.states = self.states.reshape(self.buffer_dim + self.envs.single_observation_space.shape)
            self.actions = self.actions.reshape(self.buffer_dim + self.envs.single_action_space.shape)
            self.log_probs = self.log_probs.reshape(self.buffer_dim)
            self.values = self.values.reshape(self.buffer_dim)

            # Validate the agent
            # if update % self.config.valid.n_interval == 0:
            #     self.validation(current_update=update, n_updates=rollout_update)

    def update_agent(self, states, actions, log_probs, advantages, returns):
        dataset = torch.utils.data.TensorDataset(states, actions, log_probs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.train.batch_size, shuffle=True)

        # Train the agent networks
        for _epoch in range(self.config.train.epochs):
            for b_states, b_actions, b_log_probs, b_advantages, b_returns in dataloader:
                _action, new_log_probs, entropy, new_value = self.agent.get_action_and_value(b_states, b_actions)
                log_ratio = new_log_probs - b_log_probs
                ratio = torch.exp(log_ratio)

                approximated_kl = self.compute_approximated_kl(ratio, log_ratio)

                if self.config.train.normalized_advantages:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # Eq. 7. Section 5 of PPO paper
                # The sign of the advantages, value coefficient and entropy coefficient
                # is reversed compared to the paper because of the way the optimizer works
                # (also the max function is orginialy min in the paper)
                policy_gradient_loss1 = -b_advantages * ratio
                policy_gradient_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.agent.clip_coef, 1 + self.agent.clip_coef)
                policy_gradient_loss = torch.max(policy_gradient_loss1, policy_gradient_loss2).mean()

                new_value = new_value.view(-1)
                value_loss = 0.5 * ((new_value - b_returns) ** 2).mean()

                # Eq. 9. Section 5 of PPO paper
                loss = policy_gradient_loss + self.agent.value_coef * value_loss - self.agent.entropy_coef * entropy.mean()

                self.metrics.policy_loss = numpy.append(self.metrics.policy_loss, policy_gradient_loss.item())
                self.metrics.value_loss = numpy.append(self.metrics.value_loss, value_loss.item())
                self.metrics.entropy = numpy.append(self.metrics.entropy, entropy.mean().item())
                self.metrics.loss = numpy.append(self.metrics.loss, loss.item())

                self.agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.train.max_grad_norm)
                self.agent.optimizer.step()

            # if self.config.agent.target_kl is not None and approximated_kl > self.config.agent.target_kl:
            #     break

    def compute_approximated_kl(self, ratio, log_ratio):
        with torch.no_grad():
            approximated_kl = (ratio - 1 - log_ratio).mean()
        return approximated_kl

    def validation(self, current_update: int, n_updates: int):
        self.agent.train(False)

        with torch.no_grad():
            total_reward = 0
            for _ in range(self.config.valid.n_episodes):
                state, _info = self.envs.reset()

                for _ in range(self.config.train.n_steps):
                    state = torch.tensor(state, dtype=torch.float32).to(self.device)
                    action, _, _, _ = self.agent.get_action_and_value(state)
                    state, reward, done, _, _ = self.envs.step(action.cpu().numpy())
                    total_reward += reward

                    # TODO should create its own environment for validation, total_reward might be inflated
                    # if done:
                    #     break

            total_reward = numpy.mean(total_reward) # TODO: fix (see comment above)

            self.logger.info(f"[{current_update}/{n_updates}] Validation average reward: {total_reward / self.config.valid.n_episodes}")

        self.agent.train(True)

    def plot_metrics(self, path: str):
        path = pathlib.Path(path)

        plt.figure()
        plt.plot(self.metrics.explained_variance)
        plt.xlabel("Updates")
        plt.ylabel("Explained variance")
        plt.grid()
        plt.savefig(path / "explained_variance.png")

        plt.figure()
        plt.plot(self.metrics.policy_loss)
        plt.xlabel("Updates")
        plt.ylabel("Policy loss")
        plt.grid()
        plt.savefig(path / "policy_loss.png")

        plt.figure()
        plt.plot(self.metrics.value_loss)
        plt.xlabel("Updates")
        plt.ylabel("Value loss")
        plt.grid()
        plt.savefig(path / "value_loss.png")

        plt.figure()
        plt.plot(self.metrics.entropy)
        plt.xlabel("Updates")
        plt.ylabel("Entropy")
        plt.grid()
        plt.savefig(path / "entropy.png")

        plt.figure()
        plt.plot(self.metrics.loss)
        plt.xlabel("Updates")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig(path / "loss.png")
