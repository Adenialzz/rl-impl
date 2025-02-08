import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self._buffer = deque(maxlen=buffer_size)

    def push(self, transition):
        self._buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self._buffer, batch_size)

    @property
    def size(self):
        return len(self._buffer)

class EpsilonScheduler:
    def __init__(self, init_eps: float, min_eps: float, decay_rate: float):
        self.eps = init_eps
        self.min_eps = min_eps
        self.decay_rate = decay_rate

    def get(self) -> float:
        return max(self.min_eps, self.eps)

    def step(self):
        self.eps *= self.decay_rate
