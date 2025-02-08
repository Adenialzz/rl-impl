import torch
import torch.nn as nn
import gym
from typing import Tuple, List
from simrl.schema import BatchData
from ...utils import AverageMeter, save_mlp
from ...models import MLP


class SimplePolicyGradient:
    def __init__(
        self,
        env_name: str = 'CartPole-v0',
        model_hidden_dims: List[int] = [64, 64],
        act: str = 'tanh',
        use_rtg: bool = False
    ):
        self.use_rtg = use_rtg

        self.env = gym.make(env_name)
        self.policy_model = self._build_model(self.env, model_hidden_dims, act)
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-2)

    def _build_model(self, env: gym.Env, hidden_dims: List[int], act: str) -> nn.Module:
        model = MLP(
            hidden_dims=[env.observation_space.shape[0]] + hidden_dims + [env.action_space.n],
            act = act
        )
        return model

    def get_policy(model_out_logits) -> torch.distributions.Distribution:
        policy = torch.distributions.Categorical(logits=model_out_logits)
        return policy

    @staticmethod
    @torch.no_grad()
    def get_action(policy_model: nn.Module, state: torch.Tensor) -> int:
        logits = policy_model(state)
        policy = torch.distributions.Categorical(logits=logits)
        action = policy.sample().item()
        return action


    @torch.no_grad()
    def collect_data(self, batch_size: int = 64) -> Tuple[BatchData, float, float]:
        episode_reward = []
        ep_len_stat, ep_ret_stat = AverageMeter(), AverageMeter()
        collected_data = BatchData()

        obs, _ = self.env.reset()        # obervations can be viewed as states in most cases
        while True:
            collected_data.obs.append(obs.copy())

            action = self.get_action(self.policy_model, torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, _, _  = self.env.step(action)

            collected_data.actions.append(action)
            episode_reward.append(reward)

            if done:
                curr_episode_return = sum(episode_reward)
                episode_length = len(episode_reward)

                ep_len_stat.update(episode_length)
                ep_ret_stat.update(curr_episode_return)

                if self.use_rtg:  # 使用 reward-to-go，每个动作的 return 只与其之后得到的奖励有关
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
                    obs, _ = self.env.reset()
                    done = False
                    episode_reward = []

        assert len(collected_data.actions) == len(collected_data.obs) == len(collected_data.episode_return)
        return collected_data, ep_ret_stat.avg, ep_len_stat.avg


    def update_one_step(self, data: BatchData):
        # 在收集到的数据上计算策略梯度并更新网络一步

        self.optimizer.zero_grad()
        
        obs = torch.as_tensor(data.obs, dtype=torch.float32)
        actions = torch.as_tensor(data.actions, dtype=torch.int32)
        returns = torch.as_tensor(data.episode_return, dtype=torch.float32)

        logits = self.policy_model(obs)        # 这里为啥要重新 forward 一遍算 logits ？
        policy = torch.distributions.Categorical(logits=logits)
        log_prob = policy.log_prob(actions)
        loss = - (log_prob * returns).mean()
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
            
    def run(
        self,
        total_steps: int = 50,
        batch_size: int = 5000
    ):
        for i in range(total_steps):
            data, avg_ret, avg_len = self.collect_data(batch_size)
            loss = self.update_one_step(data)
            print(f"step {i}: avg len: {avg_len:.2f}, avg return: {avg_ret:.2f}, \"loss\": {loss:.2f}")

    def save_policy_model(self, save_path: str):
        save_mlp(self.policy_model, save_path)

    def save_model(self, save_path: str):
        self.save_policy_model(save_path)


