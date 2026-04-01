import os
import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()

        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d),
        )

    def __len__(self):
        return len(self.buffer)

@dataclass
class Config:
    seed: int = 42
    total_steps: int = 100000
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 50000

    use_target: bool = False
    use_replay: bool = False

    batch_size: int = 64
    target_update: int = 1000

    hidden_sizes: tuple = (64, 64)
    updates_per_step: int = 1


class Agent:
    def __init__(self, state_dim, action_dim, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.q = QNetwork(state_dim, action_dim, cfg.hidden_sizes).to(self.device)
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)

        if cfg.use_target:
            self.target = QNetwork(state_dim, action_dim, cfg.hidden_sizes).to(self.device)
            self.target.load_state_dict(self.q.state_dict())
        else:
            self.target = None

        self.buffer = ReplayBuffer() if cfg.use_replay else None

    def epsilon(self, step):
        return max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start - step / self.cfg.epsilon_decay
        )

    def act(self, state, step):
        if random.random() < self.epsilon(step):
            return random.randint(0, 1)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.q(state)).item()

    def store(self, s, a, r, ns, d):
        if self.buffer:
            self.buffer.add(s, a, r, ns, d)
        else:
            self.last = (s, a, r, ns, d)

    def sample(self):
        if self.buffer and len(self.buffer) > self.cfg.batch_size:
            return self.buffer.sample(self.cfg.batch_size)

        s, a, r, ns, d = self.last
        return (
            np.array([s]),
            np.array([a]),
            np.array([r]),
            np.array([ns]),
            np.array([d])
        )

    def train_step(self):
        s, a, r, ns, d = self.sample()

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).unsqueeze(1).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_val = self.q(s).gather(1, a)

        with torch.no_grad():
            if self.target:
                next_q = self.target(ns).max(1)[0].unsqueeze(1)
            else:
                next_q = self.q(ns).max(1)[0].unsqueeze(1)

            target = r + self.cfg.gamma * (1 - d) * next_q

        loss = nn.MSELoss()(q_val, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

def train(cfg):
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=cfg.seed)

    agent = Agent(4, 2, cfg)

    rewards = []
    total = 0

    for step in range(cfg.total_steps):
        action = agent.act(state, step)
        ns, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(state, action, r, ns, done)

        # support update-to-data ratio
        for _ in range(cfg.updates_per_step):
            if cfg.use_replay:
                if len(agent.buffer) > cfg.batch_size:
                    agent.train_step()
            else:
                agent.train_step()

        total += r
        state = ns

        if cfg.use_target and step % cfg.target_update == 0:
            agent.target.load_state_dict(agent.q.state_dict())

        if done:
            rewards.append(total)
            total = 0
            state, _ = env.reset()

    return rewards


def plot(rewards, name):
    plt.plot(rewards)
    plt.title(name)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{name}.png")
    plt.close()


if __name__ == "__main__":
    cfg = Config(use_target=False, use_replay=False)
    rewards = train(cfg)
    plot(rewards, "naive")
    print("DONE")