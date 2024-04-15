"""
Based on https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py and
adapted to https://github.com/SherbyRobotics/pyro.git environment.
"""

import random
import os
import time

import numpy
import torch
import gymnasium as gym

import clipy
from ppo import Agent

from pyro.dynamic.boat import Boat2D


def make_env(seed, idx, capture_video, run_name):
    def thunk():
        # system = Boat2D()

        # env = system.convert_to_gymnasium()
        # env.render_mode = "rgb_array" if capture_video and idx == 0 else None

        env = gym.make("CartPole-v1", render_mode="rgb_array" if capture_video and idx == 0 else None)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def vectorize_envs(seed, n_envs, capture_video, run_name):
    envs = gym.vector.SyncVectorEnv(
        [make_env(seed, idx, capture_video, run_name) for idx in range(n_envs)]
    )

    return envs


@clipy.command()
@clipy.argument("update-epochs", help="The number of epochs to update the policy", type=int, default=10)
@clipy.argument("total-timesteps", help="The Total timesteps of the experiments", type=int, default=25000)
@clipy.argument("n-steps", help="The number of steps to run for each environment per policy rollout", type=int, default=128)
@clipy.argument("n-envs", help="The number of environments to run in parallel", type=int, default=4)
@clipy.argument("n-minibatches", help="The number of minibatches to split the batch into", type=int, default=4)
@clipy.argument("cuda", help="Use CUDA if available", action="store_true")
@clipy.argument("seed", help="The seed for the environment", type=int, default=0)
@clipy.argument("capture-video", help="Capture video of the first environment", action="store_true")
@clipy.argument("learning-rate", help="The learning rate for the optimizer", type=float, default=2.5e-4)
@clipy.argument("gamma", help="The discount factor for the returns", type=float, default=0.99)
def main(*_args, update_epochs, total_timesteps, n_steps, n_envs, n_minibatches, cuda, seed, capture_video, learning_rate, gamma, **_kwargs):    
    # RUN_NAME = f"ppo_boat_{time.strftime('%Y%m%d-%H%M%S')}"
    RUN_NAME = "ppo_boat_overwritten"

    batch_size = n_envs * n_steps
    minibatch_size = batch_size // n_minibatches

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    envs = vectorize_envs(seed, n_envs, capture_video, RUN_NAME)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    agent.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    states = torch.zeros((n_steps, n_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((n_steps, n_envs) + envs.single_action_space.shape).to(device)
    log_probs = torch.zeros((n_steps, n_envs)).to(device)
    rewards = torch.zeros((n_steps, n_envs)).to(device)
    values = torch.zeros((n_steps, n_envs)).to(device)
    dones = torch.zeros((n_steps, n_envs)).to(device)

    num_updates = total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        next_state = torch.Tensor(envs.reset()[0]).to(device)
        next_done = torch.zeros(n_envs).to(device)

        for step in range(n_steps):
            states[step] = next_state
            dones[step] = next_done

            with torch.no_grad():
                action, log_prob, _entropy, value = agent.get_action_and_value(next_state)
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            next_state, reward, next_done, _trunc, _info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).reshape(-1)
            next_state = torch.tensor(next_state).to(device)
            next_done = torch.tensor(next_done).to(device)

        # Bootstrap the value function
        with torch.no_grad():
            next_value = agent.get_value(next_state).reshape(1, -1)

            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_non_terminal = 1.0 - next_done.float()
                    next_return = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1].float()
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * next_non_terminal * next_return

            advantages = returns - values

        # Flatten the data
        states = states.reshape((-1,) + envs.single_observation_space.shape)
        actions = actions.reshape((-1,) + envs.single_action_space.shape)
        log_probs = log_probs.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        values = values.reshape(-1)

        agent.update(states, actions, log_probs, advantages, returns, update_epochs, minibatch_size)

        # Reshape the data for next iteration
        states = states.reshape((n_steps, n_envs) + envs.single_observation_space.shape)
        actions = actions.reshape((n_steps, n_envs) + envs.single_action_space.shape)
        log_probs = log_probs.reshape(n_steps, n_envs)
        values = values.reshape(n_steps, n_envs)

        # Evaluate the agent
        if update % 10 == 0:
            total_reward = 0
            total_episodes = 0

            with torch.no_grad():
                for _ in range(10):
                    state = torch.Tensor(envs.reset()[0]).to(device)
                    done = torch.zeros(n_envs).to(device)

                    for _ in range(n_steps):
                        action, _log_prob, _entropy, _value = agent.get_action_and_value(state)
                        state, reward, done, _trunc, _info = envs.step(action.cpu().numpy())
                        state = torch.tensor(state).to(device)
                        total_reward += sum(reward)
                        total_episodes += sum(done)

            print(f"Update: {update}/{num_updates}, Reward: {total_reward / total_episodes}")

    envs.close()

    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), f"models/{RUN_NAME}.pt")

    # Create videos directory
    os.makedirs("videos", exist_ok=True)


if __name__ == '__main__':
    main() # pylint: disable=missing-kwoa
