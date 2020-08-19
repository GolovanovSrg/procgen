import math
from copy import deepcopy

from gym import RewardWrapper
from gym.wrappers import FrameStack
from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper


class ProcgenEnv(RewardWrapper):
    @staticmethod
    def wrap(config):
        config = deepcopy(config)
        is_training = config.pop('is_training', False)
        gamma = config.pop('gamma', 0.9)
        frame_stack = config.pop('frame_stack', False)

        env = ProcgenEnvWrapper(config)
        if frame_stack:
            env = FrameStack(env, 4)

        return ProcgenEnv(env, gamma, is_training)

    def __init__(self, env, gamma, is_training=True):
        super().__init__(env)

        self._reward = 0
        self._gamma = gamma

        self._mean = 0
        self._m2 = 1
        self._count = 0

        self._is_training = is_training

    def _update(self, value):
        self._reward = self._gamma * self._reward + value
        value = self._reward

        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        
    def _get_stats(self):
        if self._count < 2:
            return 0, 1
        return self._mean, self._m2 / (self._count - 1)

    def reward(self, r):
        if self._is_training:
            self._update(r)
            _, var = self._get_stats()
            r /= math.sqrt(var + 1e-8)

        return r


registry.register_env("procgen_env", ProcgenEnv.wrap)
