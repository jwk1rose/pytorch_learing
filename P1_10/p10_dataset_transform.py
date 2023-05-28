import torchvision
from torch.utils.tensorboard import SummaryWriter

# 定义数据变换
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# 加载CIFAR-10数据集
train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, transform=dataset_transform, download=True)
# 创建SummaryWriter对象，并指定输出目录
writer = SummaryWriter('pic100')
# 遍历数据集，将图像添加到tensorboard中
for i in range(100):
    writer.add_image('pic', train_set[i][0], i)
# 关闭SummaryWriter对象
writer.close()
