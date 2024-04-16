"""
Based on https://github.com/vwxyzjn/ppo-implementation-details.git and
adapted to https://github.com/SherbyRobotics/pyro.git environment.
"""

import logging
import random
import os
import sys

import numpy
import torch
import gymnasium as gym

import clipy
from ppo_continuous_action import Agent

from pyro.dynamic.boat import Boat2D


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(name)s/%(levelname)s]: %(message)s", datefmt="%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

def make_env(run_name, seed, idx, capture_video=False):
    def thunk():
        system = Boat2D()

        env = system.convert_to_gymnasium()
        env.render_mode = "human" if capture_video and idx == 0 else None

        # env = gym.make("HalfCheetah-v4", render_mode="rgb_array" if capture_video and idx == 0 else None)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: numpy.clip(obs, system.x_lb, system.x_ub))
        env = gym.wrappers.TransformObservation(env, lambda obs: numpy.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(env, lambda reward: numpy.clip(reward, system.u_lb, system.u_ub))
        env = gym.wrappers.TransformReward(env, lambda reward: numpy.clip(reward, -10, 10))

        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def vectorize_envs(run_name, seed, n_envs, capture_video=False):
    envs = gym.vector.SyncVectorEnv(
        [make_env(run_name, seed + idx, idx, capture_video) for idx in range(n_envs)]
    )

    return envs

"""
python .\ppo\train.py --cuda --capture-video --n-steps 1000
"""
@clipy.command()
@clipy.argument("update-epochs", help="The number of epochs to update the policy", type=int, default=10)
@clipy.argument("total-timesteps", help="The Total timesteps of the experiments", type=int, default=2000000)
@clipy.argument("n-steps", help="The number of steps to run for each environment per policy rollout", type=int, default=2048)
@clipy.argument("n-envs", help="The number of environments to run in parallel", type=int, default=1)
@clipy.argument("n-minibatches", help="The number of minibatches to split the batch into", type=int, default=32)
@clipy.argument("cuda", help="Use CUDA if available", action="store_true")
@clipy.argument("seed", help="The seed for the environment", type=int, default=0)
@clipy.argument("capture-video", help="Capture video of the first environment", action="store_true")
@clipy.argument("learning-rate", help="The learning rate for the optimizer", type=float, default=3e-4)
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

    envs = vectorize_envs(RUN_NAME, seed, n_envs, capture_video=False)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    eval_env = make_env(RUN_NAME, seed, 0, capture_video=False)()
    assert isinstance(eval_env.action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    agent.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    states = torch.zeros((n_steps, n_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((n_steps, n_envs) + envs.single_action_space.shape).to(device)
    log_probs = torch.zeros((n_steps, n_envs)).to(device)
    rewards = torch.zeros((n_steps, n_envs)).to(device)
    values = torch.zeros((n_steps, n_envs)).to(device)
    dones = torch.zeros((n_steps, n_envs)).to(device)

    num_updates = total_timesteps // batch_size

    # logging hyperparameters
    logger.info("=========Hyperparameters=========")
    logger.info(f"Run Name: {RUN_NAME}")
    logger.info(f"update_epochs: {update_epochs}")
    logger.info(f"total_timesteps: {total_timesteps}")
    logger.info(f"n_steps: {n_steps}")
    logger.info(f"n_envs: {n_envs}")
    logger.info(f"n_minibatches: {n_minibatches}")
    logger.info(f"device: {device}")
    logger.info(f"seed: {seed}")
    logger.info(f"capture_video: {capture_video}")
    logger.info(f"learning_rate: {learning_rate}")
    logger.info(f"gamma: {gamma}")
    logger.info(f"num_updates: {num_updates}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"minibatch_size: {minibatch_size}")
    logger.info("=================================")

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
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
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

            with torch.no_grad():
                for _ in range(10): # Run 100 episodes
                    state = torch.Tensor(eval_env.reset()[0]).to(device)

                    for _ in range(n_steps):
                        action, _log_prob, _entropy, _value = agent.get_action_and_value(state)
                        state, reward, done, _trunc, _info = eval_env.step(action.cpu().numpy())
                        state = torch.tensor(state, dtype=torch.float32).to(device)
                        total_reward += reward

                        if done:
                            break

            logger.info(f"[{update}/{num_updates}] Reward: {total_reward / 10}")

    envs.close()
    eval_env.close()

    # Test the agent
    test_env = make_env(RUN_NAME, seed, 0, capture_video=capture_video)()
    assert isinstance(test_env.action_space, gym.spaces.Box), "only continuous action space is supported"

    total_reward = 0
    state = torch.Tensor(test_env.reset()[0]).to(device)
    for _ in range(n_steps):
        action, _log_prob, _entropy, _value = agent.get_action_and_value(state)
        state, reward, done, _trunc, _info = test_env.step(action.cpu().numpy())
        state = torch.tensor(state, dtype=torch.float32).to(device)

        total_reward += reward

        if done:
            break

    logger.info(f"Test Reward: {total_reward}")

    test_env.close()

    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), f"models/{RUN_NAME}.pt")

    # Create videos directory
    os.makedirs("videos", exist_ok=True)


if __name__ == '__main__':
    main() # pylint: disable=missing-kwoa
