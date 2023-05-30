import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.nn import Linear, Conv2d, MaxPool2d, Flatten

dataset = torchvision.datasets.CIFAR10("../P1_10/CIFAR10", train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

writer = SummaryWriter("CIFAR10_log")


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output


model = Model()
input = torch.ones(size=(64, 3, 32, 32))
out = model(input)
print(out.shape)
writer.add_graph(model, input)
writer.close()
