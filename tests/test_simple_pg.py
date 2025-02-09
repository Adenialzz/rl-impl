from simrl.algos.policy_gradient.simple_pg import SimplePolicyGradient
from simrl.evaluation import render, evaluation
from simrl.utils import load_mlp
from functools import partial
import os

def test_simple_policy_gradient():
    env_name='CartPole-v0'
    save_path = 'pg.pt'
    num_eval_episode = 20

    # training
    simple_pg = SimplePolicyGradient(env_name=env_name)
    simple_pg.run(total_steps=50, batch_size=5000)
    simple_pg.save_model(save_path)

    model = load_mlp(save_path)

    # format polict
    policy = partial(SimplePolicyGradient.get_action, policy_model=model)

    # render and visualize
    render(env_name, policy)
    
    # evalution
    avg_ret = evaluation(env_name, policy, num_eval_episode)
    print(f"evalution {num_eval_episode} episodes, avg return: {avg_ret}")

    os.remove(save_path)

if __name__ == '__main__':
    test_simple_policy_gradient()
