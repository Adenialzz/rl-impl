import gym
import torch
from simrl.utils import get_action
from simrl.models import MLP

device = 'cpu'

def main(
    env_name: str = 'CartPole-v0',
    model_path: str = 'CartPole-v0-step100-model.pt',
):
    # 创建环境
    env = gym.make(env_name, render_mode="human")

    model = MLP(
        in_dim=env.observation_space.shape[0],
        hidden_dim=32,
        out_dim=env.action_space.n
    ).to(device)

    # 加载模型参数
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()  # 设置为评估模式

    obs, _ = env.reset()
    done = False
    env.render()
    while not done:

        # 渲染
        env.render()

        # 前向模型获取动作
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action_logits = model(obs_tensor)
            action = get_action(action_logits)

        # 执行动作
        obs, reward, done, _, _ = env.step(action)

        if done:
            break

    env.close()

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
