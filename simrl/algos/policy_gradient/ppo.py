import gym
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from ...models import MLP
from ...utils import AverageMeter, save_mlp

@dataclass
class PPOBatchData:
    states: List[List[float]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_prob: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

class PPO:
    def __init__(
        self,
        env_name: str = 'CartPole-v0'
    ):
        self.env = gym.make(env_name)
        print(f'State Space: {self.env.observation_space.shape}, Action Space: {self.env.action_space.n}')
        self.actor = MLP(
            hidden_dims=[self.env.observation_space.shape[0]] + [256, 256] + [self.env.action_space.n], act='relu')
        self.critic = MLP(hidden_dims=[self.env.observation_space.shape[0]] + [256, 256] + [1], act='relu')

        lr = 3e-4
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.entropy_coeff = 0.01
        self.k_epochs = 4   # 更新策略网络的次数

    @staticmethod
    @torch.no_grad()
    # same with policy gradient
    def get_action(actor_model: nn.Module, state: torch.Tensor, return_log_probs = False) -> Tuple[int, Optional[torch.Tensor]]:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        # logits = actor_model(state).detach()
        logits = actor_model(state)
        policy = torch.distributions.Categorical(logits=logits)
        action = policy.sample()
        if return_log_probs:
            log_probs = policy.log_prob(action)
            return action.item(), log_probs
        else:
            return action.item()

    @torch.no_grad()
    def collect_data(self, min_data_num: int) -> Tuple[PPOBatchData, float]:
        obs, _ = self.env.reset()
        
        ep_cum_reward = 0
        return_stat = AverageMeter()

        batch_data = PPOBatchData()

        while True:
            action, logp = self.get_action(self.actor, obs, return_log_probs=True)

            next_obs, reward, done, _, _ = self.env.step(action)

            batch_data.states.append(obs)
            batch_data.actions.append(action)
            batch_data.log_prob.append(logp)
            batch_data.rewards.append(reward)
            batch_data.dones.append(done)

            ep_cum_reward += reward

            obs = next_obs

            if done:
                return_stat.update(ep_cum_reward)
                if len(batch_data.actions) > min_data_num:
                    break

                done = False
                obs, _ = self.env.reset()
                ep_cum_reward = 0

        return batch_data, return_stat.avg

    def update_one_step(self, batch_data: PPOBatchData):

        states = torch.tensor(np.array(batch_data.states), dtype=torch.float32)
        actions = torch.tensor(np.array(batch_data.actions), dtype=torch.int64)     # 就因为这里写成了 float32，就完全不 work ？？？？？？？？？？？？？改了就好了
        log_probs = torch.tensor(np.array(batch_data.log_prob), dtype=torch.float32)


        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(batch_data.rewards), reversed(batch_data.dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            

        # normalizing
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        
        # 这里更新了多次，相当于已经用重要性采样将 ppo 变成一定程度上的 "off-policy" 了
        for _ in range(self.k_epochs):
            values = self.critic(states)
            advantage = returns - values.detach()

            logits = self.actor(states)
            policy = torch.distributions.Categorical(logits=logits)
            new_log_probs = policy.log_prob(actions)

            # compute ratio (pi_theta / pi_theta_old):
            ratio = torch.exp(new_log_probs - log_probs)

            # compute surrogate loss, PPO-Clip
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            actor_loss = - torch.min(surr1, surr2).mean() + self.entropy_coeff * policy.entropy().mean()
            critic_loss = (returns - values).pow(2).mean()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()


    def run(self, total_steps: int, batch_size: int):
        for step in range(total_steps):
            batch_data, avg_ret = self.collect_data(min_data_num=batch_size)
            print(f"step: {step}, avg ret: {avg_ret }")

            self.update_one_step(batch_data)

    def save_model(self, save_path: str):
        save_mlp(self.actor, save_path)
            
