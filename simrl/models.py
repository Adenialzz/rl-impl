import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    _act_func_map = dict(
        relu=nn.ReLU,
        tanh=nn.Tanh
    )

    def __init__(
        self,
        hidden_dims: List[int],
        act: str = 'relu'
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.act = act

        act_func = self._act_func_map.get(act.lower(), None)
        if act_func is None:
            raise KeyError(f'unknown activation function: {act}, please choose from {list(self._act_func_map.keys())}')

        layers= []
        for j in range(len(hidden_dims)-1):
            layers += [nn.Linear(hidden_dims[j], hidden_dims[j+1]), act_func()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
