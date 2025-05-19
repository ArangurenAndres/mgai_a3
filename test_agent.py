import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from mapping.text_env import TextPlatformerEnv
from agents.dueling_dqn import DuelingDQN  # import from earlier
from mapping.wrapper import TorchWrapper    # import from earlier

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
NUM_EPISODES = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment setup
env = TorchWrapper(TextPlatformerEnv("levels/data/symbol/mario_1_1.txt"))
obs_shape = (1, 14, 28)
n_actions = 4

# Networks

total_reward = 0
# ε-greedy scheduling
for i in range(500):
    next_state, reward, done, _ = env.step(2)
    total_reward+=reward
    print(f"Total reward: {total_reward}")