from simrl.algos.policy_gradient.simple_pg import SimplePolicyGradient
from simrl.render import render
import os

def test_simple_policy_gradient():
    env_name='CartPole-v0'
    save_path = 'pg.pt'
    dqn = SimplePolicyGradient(env_name=env_name)


    # training
    dqn.run(total_steps=50, batch_size=5000)
    dqn.save_model(save_path)

    # render and visualize
    render('simple_policy_gradient', env_name, save_path)

    os.remove(save_path)

if __name__ == '__main__':
    test_simple_policy_gradient()
