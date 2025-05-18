import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
from text_env import TextPlatformerEnv

# Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 2.5e-4
BATCH_SIZE = 64
UPDATE_EPOCHS = 4
ROLLOUT_STEPS = 1024

Transition = namedtuple("Transition", ["state", "action", "log_prob", "reward", "done", "value"])

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh()
        )
        self.policy = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

def compute_gae(transitions, gamma=GAMMA, lam=GAE_LAMBDA):
    values = [t.value.item() for t in transitions] + [0]
    advantages, returns = [], []
    gae = 0
    for t in reversed(range(len(transitions))):
        delta = transitions[t].reward + gamma * values[t + 1] * (1 - transitions[t].done) - values[t]
        gae = delta + gamma * lam * (1 - transitions[t].done) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    return advantages, returns

def collect_rollout(env, model, device):
    obs = env.reset()
    obs = torch.tensor(obs.flatten(), dtype=torch.float32).to(device)
    transitions = []

    for _ in range(ROLLOUT_STEPS):
        obs_input = obs.unsqueeze(0)
        with torch.no_grad():
            probs, value = model(obs_input)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, done, _ = env.step(action.item())
        next_obs_tensor = torch.tensor(next_obs.flatten(), dtype=torch.float32).to(device)

        transitions.append(Transition(obs, action, log_prob, reward, done, value))

        obs = next_obs_tensor if not done else torch.tensor(env.reset().flatten(), dtype=torch.float32).to(device)
                       
    return transitions

def ppo_update(model, optimizer, transitions, device):
    states = torch.stack([t.state for t in transitions]).to(device)
    actions = torch.stack([t.action for t in transitions]).to(device)
    old_log_probs = torch.stack([t.log_prob for t in transitions]).to(device)

    advantages, returns = compute_gae(transitions)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    for _ in range(UPDATE_EPOCHS):
        for i in range(0, len(transitions), BATCH_SIZE):
            idx = slice(i, i + BATCH_SIZE)
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            probs, values = model(batch_states)
            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(batch_actions)

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (batch_returns - values.squeeze()).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_ppo(env, num_iterations=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_sample = env.reset()
    input_dim = obs_sample.size
    action_dim = env.action_space.n

    model = ActorCritic(input_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for iteration in range(num_iterations):
        transitions = collect_rollout(env, model, device)
        ppo_update(model, optimizer, transitions, device)
        total_reward = sum([t.reward for t in transitions])
        print(f"Iteration {iteration} | Total reward: {total_reward:.2f}")
    torch.save(model.state_dict(), "ppo_mario_text.pth")
if __name__ == "__main__":
    env = TextPlatformerEnv("mario_1_1.txt")
    train_ppo(env)
