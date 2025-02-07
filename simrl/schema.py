from dataclasses import dataclass, field
from typing import List

@dataclass
class BatchData:
    actions: List[int] = field(default_factory=list)
    obs: List[List[float]] = field(default_factory=list)
    episode_return: List[float] = field(default_factory=list)
