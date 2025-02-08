import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import deque
import random

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int , out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self._buffer = deque(maxlen=buffer_size)

    def push(self, transition):
        self._buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self._buffer, batch_size)

    @property
    def size(self):
        return len(self._buffer)

# 设置超参数
buffer_size = 10000
batch_size = 64
eps = 1.0 # 用于探索
eps_decay = 0.99
eps_min = 0.01
target_update_freq = 10
train_q_net_freq = 2
gamma = 0.99  # 折扣因子
total_steps = 100000

# 初始化环境
env = gym.make('CartPole-v0')

# 构建网络
net_kwargs = dict(
    in_dim=env.observation_space.shape[0],
    hidden_dim=64,
    out_dim=env.action_space.n
)

q_net = MLP(**net_kwargs)
target_net = MLP(**net_kwargs)
target_net.load_state_dict(q_net.state_dict())  # 初始化为相同的网络
optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-2)
obs, _ = env.reset()
episode_cum_reward = 0

rb = ReplayBuffer(buffer_size)

for step in range(total_steps):
    if random.random() < eps:
        # 有一定概率随机采样一个动作，用于探索其他可能性
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            logits = q_net(torch.as_tensor(obs))
        action = torch.argmax(logits).item()

    next_obs, reward, done, _, _ = env.step(action)

    rb.push(
        (obs, action, reward, next_obs, done)
    )

    obs = next_obs
    episode_cum_reward += reward

    if done:
        print(step, episode_cum_reward)

        done = False
        obs, _ = env.reset()
        episode_cum_reward = 0

    if rb.size < batch_size:  # 先积累一些数据
        continue

    if step % train_q_net_freq == 0:
        transitions = rb.sample(batch_size)
        batch = list(zip(*transitions))
        states = torch.as_tensor(batch[0], dtype=torch.float32)
        actions = torch.as_tensor(batch[1], dtype=torch.int64)
        rewards = torch.as_tensor(batch[2], dtype=torch.float32)
        next_states = torch.as_tensor(batch[3], dtype=torch.float32)
        dones = torch.as_tensor(batch[4], dtype=torch.int32)

        with torch.no_grad():
            target_q_values = target_net(next_states)  # Q(s_{t+1})
            max_next_q_values = target_q_values.max(1)[0]
            td_target = rewards + gamma * max_next_q_values * (1 - dones)  # td target

        q_values = q_net(states)
        q_values = q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_values, td_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eps = max(eps_min, eps * eps_decay)

    if step % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

torch.save(q_net.state_dict(), 'dqn.pt')
    