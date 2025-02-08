import gym
import torch
from .algos.q_learning.dqn import DQN
from .algos.policy_gradient.simple_pg import SimplePolicyGradient
from .utils import load_mlp

def render(method: str = 'dqn', env_name: str = 'CartPole-v0', model_path: str = 'model.pt'):
    _methods_map = dict(
        dqn=DQN,
        simple_policy_gradient=SimplePolicyGradient
    )
    
    env = gym.make(env_name, render_mode="human")
    method_cls = _methods_map[method.lower()]

    model = load_mlp(model_path)
    model.eval()

    obs, _ = env.reset()
    done = False
    env.render()
    episode_cum_reward = 0
    while not done:

        # 渲染
        env.render()

        # 前向模型获取动作
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action = method_cls.get_action(model, obs_tensor)

        # 执行动作
        obs, reward, done, _, _ = env.step(action)
        episode_cum_reward += reward
        print(f'cumulated reward: {episode_cum_reward}')

    env.close()
