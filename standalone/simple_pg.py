import torch
import torch.nn as nn
import gym
from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple


class MLP(nn.Module):
    def __init__(self, in_dim:int , hidden_dim:int , out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def get_policy(model_out_logits) -> torch.distributions.Distribution:
    policy = torch.distributions.Categorical(logits=model_out_logits)
    return policy

def get_action(model_out_logits) -> int:
    policy = get_policy(model_out_logits)
    action = policy.sample().item()
    return action

@dataclass
class CollectedData:
    actions: List[int] = field(default_factory=list)
    obs: List[List[float]] = field(default_factory=list)
    episode_return: List[float] = field(default_factory=list)

# @torch.no_grad()
def collect_data(env: gym.Env, model: nn.Module, use_rtg: bool = False, device: str = 'cpu') -> Tuple[CollectedData, float, float]:
    obs, _ = env.reset()        # obervations can be viewed as states in most cases
    batch_size = 5000
    episode_reward = []
    ep_lens, ep_rets = [], []

    collected_data = CollectedData()
    while True:
        collected_data.obs.append(obs.copy())

        out = model(torch.as_tensor(obs, dtype=torch.float32, device=device))
        action = get_action(out)
        obs, reward, done, _, _  = env.step(action)

        collected_data.actions.append(action)

        episode_reward.append(reward)

        if done:
            curr_episode_return = sum(episode_reward)
            episode_length = len(episode_reward)

            ep_lens.append(episode_length)
            ep_rets.append(curr_episode_return)

            if use_rtg:  # 使用 reward-to-go，每个动作的 return 只与其之后得到的奖励有关
                episode_return_per_action = [0] * episode_length
                for i in reversed(range(episode_length)):
                    episode_return_per_action[i] = episode_reward[i] + (episode_return_per_action[i+1] if i+1 < episode_length else 0)
            else:
                episode_return_per_action = [curr_episode_return] * episode_length 
            collected_data.episode_return.extend(episode_return_per_action)

            if len(collected_data.obs) > batch_size:
                break
            else:
                # 数据不够，再跑一轮
                obs, _ = env.reset()
                done = False
                episode_reward = []

    assert len(collected_data.actions) == len(collected_data.obs) == len(collected_data.episode_return)
    return collected_data, np.mean(ep_rets), np.mean(ep_lens)


def update_one_step(model: nn.Module, data: CollectedData, optimizer: torch.optim.Optimizer, device: str = 'cpu') -> float:
    # 在收集到的数据上计算策略梯度并更新网络一步

    optimizer.zero_grad()
    
    obs = torch.as_tensor(data.obs, dtype=torch.float32, device=device)
    actions = torch.as_tensor(data.actions, dtype=torch.int32, device=device)
    returns = torch.as_tensor(data.episode_return, dtype=torch.float32, device=device)

    model_out_logits = model(obs)        # 这里为啥要重新 forward 一遍算 logits ？
    policy = get_policy(model_out_logits)
    log_prob = policy.log_prob(actions)
    loss = - (log_prob * returns).mean()
    
    loss.backward()
    optimizer.step()
    return loss.item()
        
def main(
    env_name: str = 'CartPole-v0',
    lr: float = 1e-2,
    steps: int = 50,
    use_rtg: bool = False,
    save: bool = False,
    device = 'cpu'
):
    env = gym.make(env_name)

    model = MLP(
        in_dim=env.observation_space.shape[0],
        hidden_dim=32,
        out_dim=env.action_space.n
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(steps):
        data, avg_ret, avg_len = collect_data(env, model, use_rtg, device)
        loss = update_one_step(model, data, optimizer, device)
        print(f"step {i}: avg len: {avg_len:.2f}, avg return: {avg_ret:.2f}, \"loss\": {loss:.2f}")

    if save:
        torch.save(model.state_dict(), f'{env_name}-step{steps}-{'rtg-' if use_rtg else ""}model.pt')

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
