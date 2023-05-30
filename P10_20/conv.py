import torch
import torch.nn.functional as F
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 定义输入和卷积核
input = torch.tensor([[10, 10, 10],
                      [10, 10, 10],
                      [10, 10, 10]])
kernel = torch.tensor([[1, 2], [2, 1]])
# 对输入和卷积核进行形状变换
input = torch.reshape(input=input, shape=(1, 1, 3, 3))
kernel = torch.reshape(input=kernel, shape=(1, 1, 2, 2))
# 使用卷积计算输出并打印
output = F.conv2d(input, kernel, stride=1)
print("卷积输出结果：", output)
# 读取CIFAR10数据集并进行预处理
dataset = torchvision.datasets.CIFAR10("../P1_10/CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)
# 构建数据加载器
dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
# 打印数据集样本数
print("数据集样本数：", dataset.__len__())


# 定义模型
class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


# 实例化模型
model = Mymodel()
# 实例化tensorboard的SummaryWriter
writer = SummaryWriter("./conv")
# 训练模型并记录输出
step = 0
for img, label in dataloader:
    output = model(img)
    output = torch.reshape(output, shape=(-1, 3, 30, 30))
    writer.add_images('output', output, step)
    step = step + 1
# 关闭SummaryWriter
writer.close()
