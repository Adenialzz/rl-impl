from simrl.algos.q_learning.dqn import DQN
from simrl.render import render

def test_dqn():
    env_name='CartPole-v0'
    dqn = DQN(env_name=env_name)

    # training
    dqn.run(total_steps=100000, batch_size=64)
    dqn.save_q_net('dqn.pt')

    # render and visualize
    render('dqn', env_name, 'dqn.pt')

if __name__ == '__main__':
    test_dqn()
