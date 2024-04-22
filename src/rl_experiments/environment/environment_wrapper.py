import abc

import gymnasium

from rl_experiments.config import Config


class EnvironmentWrapper(abc.ABC):
    def __init__(self, config: Config):
        self.name = config.name
        self.n_envs = config.env.n_envs
        self.seed = config.env.seed
        self.record_video = config.env.record_video

        self.envs = None

        self.observation_space = None
        self.action_space = None

    def __vectorize_envs(self, n_envs: int):
        return gymnasium.vector.SyncVectorEnv(
            [self.make_env(idx, self.record_video) for idx in range(n_envs)]
        )

    def make_env(self, idx: int = 0, record_video : bool = False):
        def wrapper() -> gymnasium.Env:
            self.env = self._make_env(idx)

            env = gymnasium.wrappers.RecordEpisodeStatistics(self.env)

            if record_video and idx == 0:
                env = gymnasium.wrappers.RecordVideo(env, f"videos/{self.name}")

            # Configure the seed of the environment
            seed = self.seed + idx
            # if isinstance(env, gymnasium.Env):
            #     env.seed(seed)

            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            return env
        return wrapper

    def initialize(self):
        self.envs = self.__vectorize_envs(self.n_envs)

        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

    def get_envs(self):
        return self.envs

    def close(self):
        self.envs.close()

    @abc.abstractmethod
    def _make_env(self, idx: int = 0) -> gymnasium.Env:
        pass


class GymnasiumEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, name:str, config: Config):
        super(GymnasiumEnvironmentWrapper, self).__init__(config)

        self.name = name

    def _make_env(self, idx: int = 0) -> gymnasium.Env:
        return gymnasium.make(self.name, render_mode="rgb_array" if self.record_video and idx == 0 else None)
