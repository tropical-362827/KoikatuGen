from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    @abstractmethod
    def loss(self, *args, **kwargs) -> torch.Tensor: ...

    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError
