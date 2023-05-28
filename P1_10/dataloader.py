import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载CIFAR-10数据集并创建测试数据集的Dataloader
test_data = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=10, shuffle=True, num_workers=0, drop_last=True)
# 获取测试数据集中的第一个数据
img, target = test_data[0]
# 输出第一个数据的图像和标签（用于检查数据加载是否正确）
print(img, target)
# 创建SummaryWriter对象
writer = SummaryWriter("dataloader")
# 循环遍历10个epoch
for epoch in range(5):
    step = 0
    # 遍历测试数据集Dataloader，将图像添加到tensorboard中
    for img, target in test_loader:
        writer.add_images("epoch:{}".format(epoch), img, step)
        step += 1
    # 输出当前epoch的数值
    print(epoch)
# 关闭SummaryWriter对象
writer.close()
