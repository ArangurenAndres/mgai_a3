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
NUM_EPISODES = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment setup
env = TorchWrapper(TextPlatformerEnv("levels/data/symbol/mario_1_1.txt"))
obs_shape = (1, 14, 28)
n_actions = 4

# Networks
policy_net = DuelingDQN(obs_shape, n_actions).to(device)
target_net = DuelingDQN(obs_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=BUFFER_SIZE)

# ε-greedy scheduling
def epsilon_by_frame(frame_idx):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * frame_idx / EPS_DECAY)

# Collect initial experience
state = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = random.randint(0, n_actions - 1)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    state = next_state if not done else env.reset()

# Training loop
frame_idx = 0
for episode in range(NUM_EPISODES):
    state = env.reset()
    print("Reset state shape:", state.shape)
    total_reward = 0

    while True:
        epsilon = epsilon_by_frame(frame_idx)
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            with torch.no_grad():
                q_values = policy_net(state.unsqueeze(0).to(device))
                action = q_values.argmax().item()

        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        frame_idx += 1

        # Sample batch
        batch = random.sample(replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        # Compute Q targets
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = target_net(next_states).max(1)[0]
            targets = rewards + GAMMA * next_q_values * (~dones)

        # Loss and update
        loss = nn.MSELoss()(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Sync target network
        if frame_idx % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    print(f"Episode {episode} - Total reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

#env.env.render()
