import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input + 1
        return output


model = MyModel()
print(model(torch.tensor(1)))  # 输出： tensor(2)
