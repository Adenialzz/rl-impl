import gym
import torch
from simrl.utils import get_action, get_action_from_q
from simrl.models import MLP
from typing import Literal

device = 'cpu'

def main(
    env_name: str = 'CartPole-v0',
    method: Literal['policy_gradient', 'dqn'] = 'policy_gradient',
    model_path: str = 'CartPole-v0-step100-model.pt',
):
    # 创建环境
    env = gym.make(env_name, render_mode="human")

    model = MLP(
        hidden_dims=[env.observation_space.shape[0], 64, 64, env.action_space.n],
        act='relu'
    ).to(device)

    # 加载模型参数
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()  # 设置为评估模式

    obs, _ = env.reset()
    done = False
    env.render()
    episode_cum_reward = 0
    while not done:

        # 渲染
        env.render()

        # 前向模型获取动作
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action_logits = model(obs_tensor)
            if method == 'policy_gradient':
                action = get_action(action_logits)
            elif method == 'dqn':
                action = get_action_from_q(action_logits)
            else:
                raise ValueError(f'got not supported method {method}')

        # 执行动作
        obs, reward, done, _, _ = env.step(action)
        episode_cum_reward += reward
        print(f'final return: {episode_cum_reward}')

        if done:
            break

    env.close()

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
