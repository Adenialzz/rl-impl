import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    _act_func_map = dict(
        relu=nn.ReLU,
        tanh=nn.Tanh,
        softmax=nn.Softmax,
        no=nn.Identity
    )

    def _get_act_func(self, act_name: str) -> nn.Module:
        act_func = self._act_func_map.get(act_name.lower(), None)
        if act_func is None:
            raise KeyError(f'unknown activation function: {act_name}, please choose from {list(self._act_func_map.keys())}')

        return act_func

    def __init__(
        self,
        hidden_dims: List[int],
        act: str = 'relu',
        out_act: str = 'no'
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.act = act
        self.out_act = out_act

        act_func = self._get_act_func(act)
        out_act_func = self._get_act_func(out_act)

        layers= []
        for j in range(len(hidden_dims)-1):
            layers += [nn.Linear(hidden_dims[j], hidden_dims[j+1]), act_func()]

        layers.append(out_act_func())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
