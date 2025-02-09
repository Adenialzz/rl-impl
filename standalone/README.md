
# WARNING

个人感觉 [ppo.py](./ppo.py) 里面的实现有点问题，不应该叫 replay buffer。ppo 是 on-policy 的，应该只能用最近的策略 $\pi_\theta$ 采样出数据，并且只能顺序排列起来（而非随机采样）。应该像 [simple_pg.py](./simple_pg.py) 里那样。当然，实际上 ppo 每步采样 k_epochs 次，相当于已经通过 importance sampling 将 on-policy 转换成 "off-policy" 了，这里是叫法的一个模糊地带。但是用随机从 replay_buffer 里面采样是绝对不行的。

# Usage

Some standalone scripts is contained in this directory.

1. Start training with any algotithms:

```shell
python simple_pg.py --env_name CartPole-v0 --steps 100 --save
```

2. Render and visualize the result with trained weights:

```shell
python render.py --env_name CartPole-v0 --method policy_gradient --model_path CartPole-v0-step100-model.pt
```

(You may need to do some modifications for the network arch in render.py)
