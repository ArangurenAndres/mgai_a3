import numpy as np
import torch

class TorchWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return self._preprocess(obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return self._preprocess(next_obs), reward, done, info

    def _preprocess(self, obs):
        obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        return torch.tensor(obs, dtype=torch.float32)

    def render(self):
        self.env.render()
