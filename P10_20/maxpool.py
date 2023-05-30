import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import MaxPool2d

dataset = torchvision.datasets.CIFAR10(root="../P1_10/CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, drop_last=True)

writer = SummaryWriter(log_dir='./maxpool_log')


class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output


model = Mymodel()
step = 0
for img, label in dataloader:
    output = model(img)
    writer.add_images('maxpool', img_tensor=output, global_step=step)
    step = step + 1

writer.close()
