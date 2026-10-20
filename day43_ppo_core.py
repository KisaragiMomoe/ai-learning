"""
Day 43: 极简 PPO 核心逻辑
不依赖 trl，纯手写。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 10
STATE_DIM = 16

class MockRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(STATE_DIM + VOCAB_SIZE, 1)

    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim = -1)
        return self.fc(x)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(STATE_DIM, VOCAB_SIZE)

    def forward(self, state):
        logits = self.fc(state)
        return F.softmax(logits, dim = -1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(STATE_DIM, 1)

    def forward(self, state):
        return self.fc(state)

actor = Actor()
critic = Critic()
rm = MockRewardModel()

for param in rm.parameters():
    param.requires_grad = False

actor_optim = torch.optim.Adam(actor.parameters(), lr = 1e-3)
critic_optim = torch.optim.Adam(critic.parameters(), lr = 1e-3)

epochs = 100
lr_beta = 0.1
gamma = 0.9

for epoch in range(epochs):
    states = torch.randn(32, STATE_DIM)
    probs = actor(states)
    action_dist = torch.distributions.Categorical(probs)
    actions = action_dist.sample()
    old_log_probs = action_dist.log_prob(actions).detach()

    action_onehot = F.one_hot(actions, VOCAB_SIZE).float()
    rewards = rm(states, action_onehot).squeeze(-1)
    values = critic(states).squeeze(-1)

    advantages = rewards - values
    advantages = advantages.detach()

    current_probs = actor(states)
    current_dist = torch.distributions.Categorical(current_probs)
    new_log_probs = current_dist.log_prob(actions)

    ratio = torch.exp(new_log_probs - old_log_probs)

    epsilon = 0.2
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    critic_loss = F.mse_loss(values, rewards.detach())
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | 平均 Reward: {rewards.mean().item():.4f}")