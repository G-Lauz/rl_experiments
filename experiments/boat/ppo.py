import logging
import os
import sys

import gymnasium
import numpy

from pyro.control.reinforcementlearning import stable_baseline3_controller
from pyro.dynamic.boat import Boat2D

from rl_experiments.command import clipy
from rl_experiments.ppo import ContinuousAgent, DiscreteAgent, Trainer
from rl_experiments.config import Config
from rl_experiments.environment import EnvironmentWrapper


class NormalizedBoat2D(Boat2D):
    # Only change B actuation matrix in order to have normalized inputs values
    def B(self, q, u):
        B = numpy.zeros((3, 2))

        B[0, 0] = 8000.0
        B[1, 1] = 3000.0
        B[2, 1] = - self.l_t * 3000.0

        return B


class BoatEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, config: Config):
        super(BoatEnvironmentWrapper, self).__init__(config)

        self.system = NormalizedBoat2D()

        # Min/max state and control inputs
        self.system.x_ub = numpy.array([10, 10, 4 * numpy.pi, 10, 10, 10])
        self.system.x_lb = -self.system.x_ub

        # Min/max inputs are normalized, scalling is in the B matrix
        self.system.u_ub = numpy.array([1, 1])
        self.system.u_lb = numpy.array([-1, -1])

        # Cost function
        self.system.cost_function.Q = numpy.diag([1, 1, 6., 0.1, 0.1, 1.0])
        self.system.cost_function.R = numpy.diag([0.001, 0.001])

        # Distribution of initial states
        self.system.x0 = numpy.array([0.0, 0.0, 0.0, 0, 0, 0])
        self.x0_std = numpy.array([5.0, 5.0, 1.0, 1.0, 1.0, 0.2])

    def _make_env(self, idx: int = 0) -> gymnasium.Env:
        env = self.system.convert_to_gymnasium()
        env.reset_mode = "gaussian"
        env.x0_std = self.x0_std
        return env


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

    env_wrapper = BoatEnvironmentWrapper(config)
    env_wrapper.initialize()

    agent = None
    agent = ContinuousAgent(env_wrapper.observation_space, env_wrapper.action_space, config)
    agent.initilaize()

    trainer = Trainer(agent, env_wrapper, config, logger=logger)
    trainer.train()

    env_wrapper.close()

    os.makedirs("metrics", exist_ok=True)
    trainer.plot_metrics("metrics")

    # Save the model
    os.makedirs("models", exist_ok=True)
    agent.save(f"models/{config.name}.pt")

    # Evaluate the model
    config.env.n_envs = 1 # TODO: Patch, fix this
    eval_env_wrapper = BoatEnvironmentWrapper(config)
    eval_env_wrapper.initialize()

    ppo_controller = stable_baseline3_controller(agent)
    ppo_controller.plot_control_law(sys=eval_env_wrapper.system, n=100)

    cl_sys = ppo_controller + eval_env_wrapper.system
    cl_sys.x0 = numpy.array([-5.0, 5.0, 1.0, 2.0, 0.0, 0.0]) # Initial states
    cl_sys.compute_trajectory(tf=30.0, n=10000, solver="euler")
    cl_sys.plot_trajectory("xu")
    ani = cl_sys.animate_simulation()


if __name__ == "__main__":
    main() # pylint: disable=missing-kwoa
