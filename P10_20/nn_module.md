# torch.nn.module

## 什么是 module 它有什么用 该怎么用

`nn.Module`  是 PyTorch 中实现神经网络模型的基类，所有自定义的神经网络模型都需要继承自  `nn.Module`  类。在子类中，我们需要定义模型的结构和前向传播函数，并可以在  `__init__`  方法中定义模型的参数，例如神经网络的层数、每层的神经元数量、激活函数等等。
 使用  `nn.Module`  有以下几个好处：
 1. 统一的 API 接口：所有的神经网络模型都依赖于  `nn.Module` ，这使得 PyTorch 模型的实现具有了统一的 API，更加易于使用和扩展。
 2. 自动求导支持： PyTorch 实现了自动求导机制，而  `nn.Module`  类提供了自动求导所需的相关函数和计算图，从而简化了模型设计和训练过程。
 3. 参数管理：模型中的参数（例如权重和偏置）可以通过  `nn.Module`  类的  `parameters()`  和  `named_parameters()`  方法进行管理，从而方便地进行参数初始化和优化。
 以下是一个简单的例子，展示了如何使用  `nn.Module`  来定义一个全连接神经网络模型：


    import torch
    from torch import nn
     class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 128)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)
        def forward(self, x):
            x = x.view(-1, 784)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            return x
     net = Net()
在这个例子中，我们定义了一个全连接神经网络模型  `Net` ，它有两个隐藏层，每个隐藏层有 128 个神经元。
在  `__init__`  方法中定义了神经网络的结构， `forward`  方法定义了前向传播过程。
在  `forward`  方法中，我们将输入的数据  `x`  重新进行  `view`  操作，将其形状变为 784 维的向量，然后分别通过两个线性层和一个 ReLU 激活函数进行处理，最后输出结果。
通过继承  `nn.Module` ，我们可以使用  `net.parameters()`  方法获取模型的参数，并可以对其进行优化训练。

