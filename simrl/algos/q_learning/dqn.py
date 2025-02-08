import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from typing import List
import random
from ...models import MLP
from ...utils import save_mlp
from ...modules import ReplayBuffer, EpsilonScheduler

class DQN:
    def __init__(
        self,
        env_name: str = 'CartPole-v0',
        buffer_size: int = 10000,
        model_hidden_dims: List[int] = [64, 64],
        act: str = 'relu',
        eps: float = 1.0,
        eps_decay_rate: float = 0.99,
        eps_min: float = 0.01,
    ):

        # 初始化环境
        self.env = gym.make(env_name)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.q_net, self.target_net = self._build_model(self.env, model_hidden_dims, act)
        self._update_target_net()

        self.eps_schedule = EpsilonScheduler(init_eps=eps, min_eps=eps_min, decay_rate=eps_decay_rate)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-2)

    def _update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _build_model(self, env: gym.Env, hidden_dims: List[int], act: str):
        net_kwargs = dict(
            hidden_dims = [env.observation_space.shape[0]] + hidden_dims + [env.action_space.n],
            act=act
        )

        q_net = MLP(**net_kwargs)
        target_net = MLP(**net_kwargs)

        return q_net, target_net

    @torch.no_grad()
    def collect_init_data(self, min_data_num: int):
        obs, _ = self.env.reset()
        while True:
            if random.random() < self.eps_schedule.get():
                # 有一定概率随机采样一个动作，用于探索其他可能性
                action = self.env.action_space.sample()
            else:
                self.get_action(self.q_net, torch.as_tensor(obs, dtype=torch.float32))

            next_obs, reward, done, _, _ = self.env.step(action)

            self.replay_buffer.push(
                (obs, action, reward, next_obs, done)
            )

            obs = next_obs

            if done:
                if self.replay_buffer.size > min_data_num:
                    return

                done = False
                obs, _ = self.env.reset()
                episode_cum_reward = 0

    @staticmethod
    @torch.no_grad()
    def get_action(q_net: nn.Module, state: torch.Tensor) -> int:
        q_values = q_net(state)
        action = torch.argmax(q_values).item()
        return action

    def run(self, total_steps: int = 50000, batch_size: int = 64, gamma: float = 0.99, target_update_freq: int = 10):
        self.collect_init_data(min_data_num=batch_size)

        obs, _ = self.env.reset()
        episode_cum_reward = 0
        for step in range(total_steps):
            # 1. 走一步，更新一个数据到 replay buffer
            # 2. 从缓存取 batch_size 个数据，更新一步 q_net
            # 3. 按照 freq 更新 target_net


            if random.random() < self.eps_schedule.get():
                # 有一定概率随机采样一个动作，用于探索其他可能性
                action = self.env.action_space.sample()
            else:
                action = self.get_action(self.q_net, torch.as_tensor(obs, dtype=torch.float32))

            next_obs, reward, done, _, _ = self.env.step(action)

            self.replay_buffer.push(
                (obs, action, reward, next_obs, done)
            )

            obs = next_obs
            episode_cum_reward += reward

            if done:
                print(f"step: {step} | curr episode return: {int(episode_cum_reward)}")

                done = False
                obs, _ = self.env.reset()
                episode_cum_reward = 0

            # 从 replay buffer 取数据并更新一步 qnet
            transitions = self.replay_buffer.sample(batch_size)
            batch = list(zip(*transitions))

            states = torch.as_tensor(batch[0], dtype=torch.float32)
            actions = torch.as_tensor(batch[1], dtype=torch.int64)
            rewards = torch.as_tensor(batch[2], dtype=torch.float32)
            next_states = torch.as_tensor(batch[3], dtype=torch.float32)
            dones = torch.as_tensor(batch[4], dtype=torch.int32)

            with torch.no_grad():
                target_q_values = self.target_net(next_states)  # Q(s_{t+1})
                max_next_q_values = target_q_values.max(1)[0]
                td_target = rewards + gamma * max_next_q_values * (1 - dones)  # td target

            q_values = self.q_net(states)
            q_values = q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

            loss = F.mse_loss(q_values, td_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.eps_schedule.step()

            if step % target_update_freq == 0:
                self._update_target_net()

    def save_q_net(self, save_path: str):
        save_mlp(self.q_net, save_path)

    def save_model(self, save_path: str):
        self.save_q_net(save_path)