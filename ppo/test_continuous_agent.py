import numpy
import torch
import gymnasium as gym

from ppo_continuous_action import Agent

from pyro.dynamic.boat import Boat2D
from pyro.control.reinforcementlearning import stable_baseline3_controller

def main():
    system = Boat2D()
    env = system.convert_to_gymnasium()
    env.render_mode = "human"

    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: numpy.clip(obs, system.x_lb, system.x_ub))
    # env = gym.wrappers.NormalizeReward(env)

    env = gym.vector.SyncVectorEnv([lambda: env])

    agent = Agent(env)

    ppo_ctl = stable_baseline3_controller(agent)
    ppo_ctl.plot_control_law(sys=system, n=100)

    # Animating rl closed-loop
    cl_sys = ppo_ctl + system
    cl_sys.x0 = numpy.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    cl_sys.compute_trajectory(tf=10.0, n=10000, solver="euler")
    cl_sys.plot_trajectory("xu")
    cl_sys.animate_simulation()

if __name__ == "__main__":
    main()
