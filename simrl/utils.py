import torch
from typing import Optional
from .models import MLP


def save_mlp(model: MLP, save_path: str):
    mlp_config = dict(
        hidden_dims=model.hidden_dims,
        act=model.act
    )

    ckpt = dict(
        mlp_config=mlp_config,
        state_dict=model.state_dict()
    )

    torch.save(ckpt, save_path)

def load_mlp(model_path: str) -> MLP:
    ckpt = torch.load(model_path, map_location='cpu')
    model = MLP(**ckpt['mlp_config'])
    model.load_state_dict(ckpt['state_dict'])
    return model

class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''

    def __init__(self, digits: Optional[int] = None):
        self.reset()
        self._digits = digits

    def reset(self):
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._sum += val * n
        self._count += n
        self._avg = self.sum / self.count
        
    @property
    def avg(self):
        if self._digits is not None:
            return round(self._avg, self._digits)
        else:
            return self._avg

    @property
    def sum(self):
        if self._digits is not None:
            return round(self._sum, self._digits)
        else:
            return self._sum
    
    @property
    def count(self):
        return self._count
