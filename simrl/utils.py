import torch
from typing import Optional

def get_policy(model_out_logits) -> torch.distributions.Distribution:
    policy = torch.distributions.Categorical(logits=model_out_logits)
    return policy

def get_action(model_out_logits) -> int:
    policy = get_policy(model_out_logits)
    action = policy.sample().item()
    return action


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
