from collections import defaultdict

import numpy as np
import torch


class LogMetrics:
    @staticmethod
    def L0(batch_tensor: torch.Tensor) -> torch.Tensor:
        return torch.count_nonzero(batch_tensor, dim=-1).float().mean()

    def __init__(self):
        self._init()

    def _init(self):
        self.metrics = defaultdict(list)

    def clear(self):
        self._init()

    def add(self, key: str, value: float | torch.Tensor):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().item()
        self.metrics[key].append(float(value))

    def add_dict(self, metrics: dict[str, float | torch.Tensor]):
        for key, value in metrics.items():
            self.add(key, value)

    def _process(self, np_func):
        return {key: float(np_func(values)) for key, values in self.metrics.items()}

    def mean(self):
        return self._process(np.mean)

    def max(self):
        return self._process(np.max)

    def min(self):
        return self._process(np.min)

    def std(self):
        return self._process(np.std)

    def median(self):
        return self._process(np.median)
