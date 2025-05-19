# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# from mapping.text_env import TextPlatformerEnv
# from agents.dueling_dqn import DuelingDQN  # import from earlier
# from mapping.wrapper import TorchWrapper    # import from earlier

# # Hyperparameters
# GAMMA = 0.99
# LR = 1e-4
# BATCH_SIZE = 32
# BUFFER_SIZE = 10000
# MIN_REPLAY_SIZE = 1000
# EPS_START = 1.0
# EPS_END = 0.1
# EPS_DECAY = 10000
# TARGET_UPDATE_FREQ = 1000
# NUM_EPISODES = 5000

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Environment setup
# env = TorchWrapper(TextPlatformerEnv("levels/data/symbol/mario_1_1.txt"))
# obs_shape = (1, 14, 28)
# n_actions = 4

# # Networks
# policy_net = DuelingDQN(obs_shape, n_actions).to(device)
# target_net = DuelingDQN(obs_shape, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())

# optimizer = optim.Adam(policy_net.parameters(), lr=LR)
# replay_buffer = deque(maxlen=BUFFER_SIZE)

# # ε-greedy scheduling
# def epsilon_by_frame(frame_idx):
#     return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * frame_idx / EPS_DECAY)

# # Collect initial experience
# state = env.reset()
# for _ in range(MIN_REPLAY_SIZE):
#     action = random.randint(0, n_actions - 1)
#     next_state, reward, done, _ = env.step(action)
#     replay_buffer.append((state, action, reward, next_state, done))
#     state = next_state if not done else env.reset()

# # Training loop
# frame_idx = 0

# for episode in range(NUM_EPISODES):
#     highest_reward = -float('inf')
#     state = env.reset()
#     total_reward = 0
    
#     max_x = 0
#     while True:
        
#         epsilon = epsilon_by_frame(frame_idx)
#         if random.random() < epsilon:
#             action = random.randint(0, n_actions - 1)
#         else:
#             with torch.no_grad():
#                 q_values = policy_net(state.unsqueeze(0).to(device))
#                 action = q_values.argmax().item()

#         next_state, reward, done, _ = env.step(action)
#         replay_buffer.append((state, action, reward, next_state, done))
#         state = next_state
#         total_reward += reward
#         frame_idx += 1

#         # Sample batch
#         batch = random.sample(replay_buffer, BATCH_SIZE)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         states = torch.stack(states).to(device)
#         actions = torch.tensor(actions, dtype=torch.long).to(device)
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
#         next_states = torch.stack(next_states).to(device)
#         dones = torch.tensor(dones, dtype=torch.bool).to(device)

#         # Compute Q targets
#         q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#         with torch.no_grad():
#             next_q_values = target_net(next_states).max(1)[0]
#             targets = rewards + GAMMA * next_q_values * (~dones)

#         # Loss and update
#         loss = nn.MSELoss()(q_values, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Sync target network
#         if frame_idx % TARGET_UPDATE_FREQ == 0:
#             target_net.load_state_dict(policy_net.state_dict())
#         if done:
#             break
        

#     print(f"Episode {episode} - Total reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")
#     env.env.render()
#     if total_reward > highest_reward:
#         highest_reward = total_reward
#         torch.save(policy_net.state_dict(), "dueldqn.pth")

# #env.env.render()
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import time
from mapping.text_env import TextPlatformerEnv
from agents.dueling_dqn import DuelingDQN
from mapping.wrapper import TorchWrapper

# Hyperparameters
GAMMA = 0.99
LR = 3e-4  # Increased learning rate
BATCH_SIZE = 64  # Larger batch size
BUFFER_SIZE = 100000  # Much larger buffer
MIN_REPLAY_SIZE = 5000  # More initial experience
EPS_START = 1.0
EPS_END = 0.01  # Lower end epsilon
EPS_DECAY = 100000  # Much slower decay
TARGET_UPDATE_FREQ = 1000
NUM_EPISODES = 5000  # More episodes
SAVE_FREQ = 100  # Save frequency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create logs directory
os.makedirs("logs", exist_ok=True)

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

# Load existing model if available
if os.path.exists("dueldqn.pth"):
    print("Loading existing model...")
    policy_net.load_state_dict(torch.load("dueldqn.pth", map_location=device))
    target_net.load_state_dict(policy_net.state_dict())

# ε-greedy scheduling
def epsilon_by_frame(frame_idx):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * frame_idx / EPS_DECAY)

# Collect initial experience
print(f"Collecting {MIN_REPLAY_SIZE} initial experiences...")
state = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = random.randint(0, n_actions - 1)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    state = next_state if not done else env.reset()

# Training loop
frame_idx = 0
highest_reward = -float('inf')
episode_rewards = []
progress_log = []

print("Starting training...")
start_time = time.time()

for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    max_x = 0
    steps = 0
    
    while True:
        steps += 1
        epsilon = epsilon_by_frame(frame_idx)
        
        # Exploration-exploitation
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            with torch.no_grad():
                q_values = policy_net(state.unsqueeze(0).to(device))
                action = q_values.argmax().item()

        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Track max position
        if 'position' in info:
            max_x = max(max_x, info['position'][0])
        
        # Store transition
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        frame_idx += 1

        # Learning
        if len(replay_buffer) >= BATCH_SIZE:
            # Sample batch
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.stack(next_states).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)

            # Double DQN: use policy net to select actions, target net for value estimation
            with torch.no_grad():
                next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
                next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
                targets = rewards + GAMMA * next_q_values * (~dones)

            # Current predictions
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Huber loss for more stable learning with outliers
            loss = nn.SmoothL1Loss()(q_values, targets)
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
            optimizer.step()

        # Sync target network
        if frame_idx % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if done:
            break
    
    episode_rewards.append(total_reward)
    
    # Log progress
    avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    progress_log.append({
        'episode': episode,
        'reward': total_reward, 
        'avg_reward': avg_reward,
        'epsilon': epsilon,
        'max_position': max_x,
        'steps': steps
    })
    
    # Print status
    if episode % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Episode {episode}/{NUM_EPISODES} - Reward: {total_reward:.2f} - Avg100: {avg_reward:.2f} "
              f"- Epsilon: {epsilon:.3f} - Max pos: {max_x} - Steps: {steps} - Time: {elapsed:.1f}s")
        env.env.render()
    # Save best model
    if total_reward > highest_reward:
        highest_reward = total_reward
        torch.save(policy_net.state_dict(), "best_dueldqn.pth")
        #print(f"New best model with reward {highest_reward:.2f}")
    
    # Periodic save
    if episode % SAVE_FREQ == 0:
        #torch.save(policy_net.state_dict(), f"dueldqn_ep{episode}.pth")
        
        # Save progress log
        with open(f"logs/progress_log_ep{episode}.txt", 'w') as f:
            for entry in progress_log:
                f.write(f"{entry}\n")
    
# Final save
torch.save(policy_net.state_dict(), "final_dueldqn.pth")
print(f"Training complete. Best reward: {highest_reward:.2f}")

# Render final trained agent
print("Rendering trained agent...")
env.env.reset()
state = env.reset()
policy_net.load_state_dict(torch.load("best_dueldqn.pth"))
policy_net.eval()

total_reward = 0
done = False

while not done:
    env.env.render()
    with torch.no_grad():
        q_values = policy_net(state.unsqueeze(0).to(device))
        action = q_values.argmax().item()
    
    state, reward, done, _ = env.step(action)
    total_reward += reward
    time.sleep(0.1)  # Slow rendering for visibility

print(f"Final evaluation reward: {total_reward:.2f}")