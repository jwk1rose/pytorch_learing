# Dataloader

## 什么是dataloader，它有什么用处，该如何使用
>DataLoader是PyTorch中用来加载数据的工具类。它可以将数据集按照指定的batch_size划分成小的batch，并对数据进行批量处理，提高了数据加载的效率。
 使用DataLoader需要先通过 `torch.utils.data.Dataset` 创建一个数据集，然后将数据集传递给DataLoader类，同时指定batch_size、shuffle和num_workers等参数，最终创建DataLoader对象。
 使用DataLoader可以方便地对数据集进行遍历，避免了手动加载数据并进行批量处理的繁琐过程，可以大大提高代码的可读性和编写效率。而在深度学习中，数据集是训练和测试的基础，因此DataLoader也是深度学习中必不可少的工具之一。
 下面是一个简单的使用DataLoader的示例代码：


    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
     # 创建自定义数据集
    class MyDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform
         def __getitem__(self, index):
            img, target = self.data[index]
            if self.transform:
                img = self.transform(img)
            return img, target
         def __len__(self):
            return len(self.data)
     # 加载MNIST数据集
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
     # 创建数据集的Dataloader
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
     # 遍历数据集的Dataloader
    for imgs, targets in train_loader:
        # ...
        pass

在这个示例中，我们自定义了一个数据集 `MyDataset` ，并使用 `DataLoader` 将其转换成Dataloader。在训练和测试时，我们可以直接使用Dataloader进行遍历，省去了手动加载数据和批处理的过程，非常方便。


## 自定义数据集如何使用dataloader
使用DataLoader加载自己的数据集主要需要以下几个步骤：
 1. 定义自定义的数据集类，继承自 `torch.utils.data.Dataset` 。需要实现  `__getitem__`  和  `__len__`  两个方法。
 2. 在自定义数据集类中实现数据预处理，例如缩放、裁剪、旋转等操作，可以使用 PyTorch 内置的图像变换模块  `torchvision.transforms` 。
 3. 创建自定义数据集对象，并传递给 DataLoader 对象，同时指定 batch_size 等参数。
 4. 遍历 DataLoader 对象，对数据进行训练或测试。
 下面是一个示例代码，用于说明如何使用 DataLoader 加载自己的数据集：


    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
     # 自定义数据集类
    class MyDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = data
            self.targets = targets
            self.transform = transform
         def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            if self.transform:
                x = self.transform(x)
            return x, y
         def __len__(self):
            return len(self.data)
     # 加载自定义数据集
    train_data = [...]  # 自己的训练数据
    train_targets = [...]  # 自己的训练标签
    train_set = MyDataset(train_data, train_targets,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5], std=[0.5])
                          ]))
     test_data = [...]  # 自己的测试数据
    test_targets = [...]  # 自己的测试标签
    test_set = MyDataset(test_data, test_targets,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5], std=[0.5])
                         ]))
     # 创建 DataLoader 对象
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
     # 遍历 DataLoader 对象
    for inputs, targets in train_loader:
        # 模型训练代码
        pass
     for inputs, targets in test_loader:
        # 模型测试代码
        pass
在这个示例中，我们定义了一个自定义数据集类  `MyDataset` ，同时使用  `DataLoader`  对象加载训练和测试数据集。
在自定义数据集类中，我们实现了  `__getitem__`  和  `__len__`  两个方法，并使用  `torchvision.transforms`  对数据进行了预处理。
然后将数据集对象传递给  `DataLoader`  对象，设置 batch_size 和 shuffle 等参数，最后遍历  `DataLoader`  对象进行训练或测试。
