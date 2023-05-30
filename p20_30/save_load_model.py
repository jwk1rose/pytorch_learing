import torch
import torchvision
from torch import nn
from torchvision import models

vgg16_true = models.vgg16(weights='IMAGENET1K_V1')
# 保存方式一 模型结构加模型参数都保存

torch.save(vgg16_true, "vgg16_1.pth")

# 保存方式二，模型参数 (官方推荐)

torch.save(vgg16_true.state_dict(), "vgg16_2.pth")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=10, out_features=1)

    def forward(self, input):
        x = self.linear(input)
        return x


model = Model()
torch.save(model.state_dict(), "my_model.pth")

# 加载

# 方式一
load_vgg16_1 = torch.load("vgg16_1.pth")
# 错误的 因为vgg16_2只保存了参数没有保存网络的结构
load_vgg16_2 = torch.load("vgg16_2.pth")
# 方式二
load_vgg16_3 = models.vgg16()
load_vgg16_3.load_state_dict(torch.load("vgg16_2.pth"))
print(load_vgg16_3)
