import gym
import torch
from typing import Callable, Tuple
from .utils import AverageMeter

@torch.no_grad()
def render(
    env_name: str,
    policy: Callable[[Tuple[float]], int],  # state ---> action
):
    env = gym.make(env_name, render_mode="human")

    state, _ = env.reset()
    done = False
    env.render()
    episode_cum_reward = 0
    while not done:

        # 渲染
        env.render()

        # 前向模型获取动作
        action = policy(state=state)

        # 执行动作
        state, reward, done, _, _ = env.step(action)
        episode_cum_reward += reward

    print(f'cumulated reward: {episode_cum_reward}')
    env.close()



@torch.no_grad()
def evaluation(  # discrete action space
    env_name: str,
    policy: Callable[[Tuple[float]], int],  # get action from state
    num_episodes: int = 10
) -> float:
    env = gym.make(env_name)

    return_stat = AverageMeter()

    state, _ = env.reset()

    ep_cum_reward = 0
    while num_episodes > 0:
        action = policy(state=state)

        state, reward, done, _, _ = env.step(action)
        ep_cum_reward += reward

        if done:
            return_stat.update(ep_cum_reward)
            num_episodes -= 1

            state, _ = env.reset()
            ep_cum_reward = 0

    return return_stat.avg
