# 加速收敛方法

## 数据归一化

## adam优化器

## 自适应学习率调整

## 模型内加入batchnorm

## torch中自动变学习率的函数要怎么使用

[>]()
在PyTorch中，可以使用torch.optim.lr_scheduler模块来实现自动变学习率的函数。常见的学习率调度方式包括StepLR、ReduceLROnPlateau和CosineAnnealingLR等。这些调度器可以根据训练的epoch数或验证集的表现自动调整学习率，从而提高模型的训练效果。
以StepLR为例，我们可以使用以下代码来实现自动变学习率的功能：

    import torch
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler
    model = MyModel()  # 构建模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 定义学习率调度器
    for epoch in range(num_epochs):
        train()  # 训练模型
        scheduler.step()  # 更新学习率
        validate()  # 验证模型

在上述代码中，我们首先定义了一个Adam优化器，并将初始学习率设置为0.001。然后，我们定义了一个StepLR学习率调度器，其中step_size参数表示每隔多少个epoch调整一次学习率，gamma参数表示学习率的下降因子。在每个epoch结束后，我们调用scheduler.step()
函数来更新学习率，然后进行训练和验证。
对于其他学习率调度方式，只需要更改调度器的类型和参数即可。需要注意的是，在使用学习率调度器时，我们通常不需要手动设置学习率，而是让调度器自动调整学习率，从而提高模型的训练效果。

## 为什么反向传播之前需要清空梯度

>在神经网络中，我们使用反向传播算法来计算梯度并更新模型参数。在进行反向传播之前，我们需要将之前计算的梯度清空，否则会将之前的梯度与当前梯度相加，从而导致梯度计算错误。因此，我们在每次反向传播之前都需要使用optimizer.zero_grad()
将梯度缓存清零，以确保梯度计算的准确性。