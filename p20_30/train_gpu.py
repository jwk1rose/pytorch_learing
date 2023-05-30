import torch.optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import time
from torch import nn
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, BatchNorm2d

device = torch.device("cpu")


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(num_features=32),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            BatchNorm2d(num_features=32),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            BatchNorm2d(num_features=64),

            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output


writer = SummaryWriter(log_dir='./logs')
train_data = torchvision.datasets.CIFAR10(root="../data", train=True,
                                          transform=torchvision.transforms.Compose(
                                              [torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5])]),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False,
                                         transform=torchvision.transforms.Compose(
                                             [torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])]), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为:{},测试数据集长度为:{}".format(train_data_size, test_data_size))

batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
model = Model()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')  # 定义学习率调度器

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数

epochs = 100
for epoch in range(epochs):
    start = time.time()
    print("---------------第{}轮训练开始----------------".format(epoch))
    model.train()

    for img, target in train_dataloader:
        img = img.to(device)
        target = target.to(device)
        output = model(img)
        loss = loss_fn(output, target)

        # 优化器
        optimizer.zero_grad()
        loss.backward()
        scheduler.step(metrics=loss)

        total_train_step += 1
        if total_train_step % 200 == 0:
            print("训练次数: {},loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar(tag='loss', scalar_value=loss, global_step=total_test_step)

    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img = img.to(device)
            target = target.to(device)
            output = model(img)
            loss = loss_fn(output, target)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的平均正确率:{}".format(total_accuracy / (test_data_size)))
    end = time.time()
    print("cost_time{}".format(end - start))
    writer.add_scalar(tag='test_loss', scalar_value=total_test_loss, global_step=total_test_step)
    writer.add_scalar(tag='test_accuracy', scalar_value=total_accuracy / (test_data_size),
                      global_step=total_test_step)
    total_test_step += 1
    if epoch % 10 == 0 and epoch >= 10:
        torch.save(model.state_dict(), "./weights/model_{}.pth".format(epoch))
        print('模型已经保存')

writer.close()
