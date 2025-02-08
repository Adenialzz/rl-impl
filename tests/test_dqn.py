from simrl.algos.q_learning.dqn import DQN
from simrl.render import render
import os

def test_dqn():
    env_name='CartPole-v1'
    dqn = DQN(env_name=env_name)

    save_path = 'dqn.pt'

    # training
    dqn.run(total_steps=100000, batch_size=64)
    dqn.save_model(save_path)

    # render and visualize
    render('dqn', env_name, save_path)

    os.remove(save_path)

if __name__ == '__main__':
    test_dqn()
