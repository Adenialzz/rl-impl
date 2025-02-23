# Reinforcement Learning Implementations

# Installation

Install `simrl` package with the following commands:

```shell
git clone git@github.com:Adenialzz/rl-impl.git
cd rl-impl
pip install -e .
```

## Simple RL training

- 

## Play with Stable Baselines 3

simrl only test on the easist RL task (CartPole). For more stable training on more difficult RL tasks, one can consider training with [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and simple testing and rendering with simrl.

training:

```python
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

env_name = "LunarLander-v2"
env = gym.make(env_name)

model = DQN(
    "MlpPolicy", 
    env=DummyVecEnv([lambda : env])
    learning_rate=5e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    target_update_interval=250,
    policy_kwargs={"net_arch" : [256, 256]},
    verbose=1,
    tensorboard_log="./tensorboard/LunarLander-v2/"
)

model.learn(total_timesteps=1e6)

model.save('lunarlander.pkl')
```

and rendering:

```python
from simrl.evaluation import render

model = DQN.load("./lunarlander.pkl")

def policy_with_sb3_model(state):
    action, _ = model.predict(observation=state)
    return action

render(env_name=env_name, policy=policy_with_sb3_model)
```


# Standalone Scripts

Some basic rl algorithms is implemented in a standalone manner (without depending on `simrl` package) in [./standalone](./standalone) directory.
