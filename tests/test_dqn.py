from simrl.algos.q_learning.dqn import DQN
from simrl.evaluation import render, evaluation
from functools import partial
import os

def test_dqn():
    env_name='CartPole-v1'
    dqn = DQN(env_name=env_name)

    save_path = 'dqn.pt'

    # training
    dqn.run(total_steps=20000, batch_size=64)
    dqn.save_model(save_path)

    policy = partial(DQN.get_action, q_net=dqn.q_net)

    # render and visualize
    render(env_name, policy=policy)

    avg_ret = evaluation(env_name, policy, 10)
    print("avg return: ", avg_ret)

    os.remove(save_path)

if __name__ == '__main__':
    test_dqn()
