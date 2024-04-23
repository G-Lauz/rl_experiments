import abc
from logging import Logger

import numpy
import torch

from rl_experiments.config import Config
from rl_experiments.environment import EnvironmentWrapper
from rl_experiments.ppo import Agent


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
                next_value = self.agent.get_value(next_state).reshape(1, -1)

                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.config.train.n_steps)):
                    if t == self.config.train.n_steps - 1:
                        next_non_terminal = 1.0 - next_done.float()
                        next_return = next_value
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

            # Reshape the data for next iteration
            self.states = self.states.reshape(self.buffer_dim + self.envs.single_observation_space.shape)
            self.actions = self.actions.reshape(self.buffer_dim + self.envs.single_action_space.shape)
            self.log_probs = self.log_probs.reshape(self.buffer_dim)
            self.values = self.values.reshape(self.buffer_dim)

            # Validate the agent
            if update % self.config.valid.n_interval == 0:
                self.validation(current_update=update, n_updates=rollout_update)

    def update_agent(self, states, actions, log_probs, advantages, returns):
        dataset = torch.utils.data.TensorDataset(states, actions, log_probs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.train.batch_size, shuffle=True)

        # Train the agent networks
        for _epoch in range(self.config.train.epochs):
            for b_states, b_actions, b_log_probs, b_advantages, b_returns in dataloader:
                _action, new_log_probs, entropy, new_value = self.agent.get_action_and_value(b_states, b_actions)
                log_ration = new_log_probs - b_log_probs
                ratio = torch.exp(log_ration)

                # Eq. 7. Section 5 of PPO paper
                # The sign of the advantages, value coefficient and entropy coefficient
                # is reversed compared to the paper because of the way the optimizer works
                # (also the max function is orginialy min in the paper)
                policy_gradient_loss1 = -b_advantages * ratio
                policy_gradient_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.agent.clip_coef, 1 + self.agent.clip_coef)
                policy_gradient_loss = torch.max(policy_gradient_loss1, policy_gradient_loss2).mean()

                value_loss = 0.5 * ((b_returns - new_value) ** 2).mean()

                # Eq. 9. Section 5 of PPO paper
                loss = policy_gradient_loss + self.agent.value_coef * value_loss - self.agent.entropy_coef * entropy.mean()

                self.agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.train.max_grad_norm)
                self.agent.optimizer.step()

    def validation(self, current_update: int, n_updates: int):
        self.agent.train(False)

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

