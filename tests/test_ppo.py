from simrl.algos.policy_gradient.ppo import PPO
from simrl.evaluation import render, evaluation

from functools import partial
import os

def test_ppo():
    env_name='CartPole-v1'
    ppo = PPO(env_name=env_name)

    save_path = 'ppo.pt'

    # training
    ppo.run(total_steps=50, batch_size=5000)
    ppo.save_model(save_path)

    # render and visualize
    policy = partial(PPO.get_action, actor_model=ppo.actor)

    render(env_name, policy)

    avg_ret = evaluation(env_name, policy)
    print("avg return: ", avg_ret)

    os.remove(save_path)

if __name__ == '__main__':
    test_ppo()
