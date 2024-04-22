import logging
import os
import sys

import torch

from rl_experiments.command import clipy
from rl_experiments.ppo import DiscreteAgent
from rl_experiments.config import Config
from rl_experiments.environment import GymnasiumEnvironmentWrapper
from rl_experiments.ppo import Trainer

def configure_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(name)s/%(levelname)s]: %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


@clipy.command()
@clipy.argument("config", help="Path to the configuration file", required=True, type=str)
def main(config):
    config = Config(config)

    logger = configure_logging()

    env_wrapper = GymnasiumEnvironmentWrapper("CartPole-v1", config)
    env_wrapper.initialize()

    agent = DiscreteAgent(env_wrapper.observation_space, env_wrapper.action_space, config)
    agent.initilaize()

    trainer = Trainer(agent, env_wrapper, config, logger=logger)
    trainer.train()

    env_wrapper.close()

    # Save the model
    os.makedirs("models", exist_ok=True)
    agent.save(f"models/{config.name}.pt")

    # Create videos directory
    os.makedirs("videos", exist_ok=True)

    # Evaluate the model
    env_wrapper.record_video = True
    test_env = env_wrapper.make_env(0, record_video=True)() # TODO: Patch, fix this
    device = torch.device("cuda" if config.train.cuda else "cpu")

    total_reward = 0
    state, _ = test_env.reset()
    for _ in range(config.eval.n_steps):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action, _, _, _ = agent.get_action_and_value(state)
        state, reward, done, _, _ = test_env.step(action.cpu().numpy())

        total_reward += reward

        if done:
            break

    logger.info(f"Test Reward: {total_reward}")

    test_env.close()


if __name__ == "__main__":
    main() # pylint: disable=missing-kwoa
