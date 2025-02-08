
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
