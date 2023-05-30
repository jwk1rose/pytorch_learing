from torch import nn
from torch.nn import Sigmoid
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import torchvision

dataset = torchvision.datasets.CIFAR10(root="../P1_10/CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

writer = SummaryWriter(log_dir="./nonliner_log")


class Model(nn.Module):
    # 在子类中，如果我们想要覆盖父类的某些方法，但仍然希望保留父类的一些属性和方法，就需要在子类中调用父类的初始化方法。
    # 这样可以确保子类中继承了父类的实例属性和方法，并且可以在子类中对其进行修改和扩展。
    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


model = Model()
step = 0
for img, table in dataloader:
    output = model(img)
    step = step + 1
    print(output.shape)
    writer.add_image('output', output, step, dataformats='nCHW')
    # writer.add_image('input', img, step)

writer.close()
