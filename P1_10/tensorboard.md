# TensorBoard

> 使用 TensorBoard 会影响训练速度嘛？
>>使用 TensorBoard 不会直接影响模型的训练速度。TensorBoard 使用事件文件来可视化模型的训练过程和性能指标，
这些事件文件是在模型训练过程中异步写入的。因此，TensorBoard
运行的时间不会影响模型的训练速度。但是，在某些情况下，如果事件数据量过大，可能会对磁盘 I/O
产生一定的影响，从而导致训练速度略微变慢。但是，这种影响通常比较小，不会对模型训练产生太大影响。

>生成log以后在当前项目的工作空间路径下输入`tensorboard --logdir=P1_10/logs --port=6007`
> - `--logdir`是log文件的路径
> - `--port` 是要打开的端口号
>才可以打开tensorboard使用
## 使用SummaryWriter将数据添加到TensorBoard可视化
 以下代码演示了如何使用SummaryWriter将图像和标量值添加到TensorBoard可视化：

        from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter
        from PIL import Image  # 导入Image类
        import numpy as np  # 导入numpy库
         writer = SummaryWriter("logs")  # 实例化SummaryWriter对象
        image_path = 'Dataset/train/ants_image/0013035.jpg'  # 图像文件路径
        image = Image.open(image_path)  # 打开图像文件
        image = np.array(image)  # 将图像文件转换成numpy数组
        writer.add_image('img', image, global_step=0,
                         dataformats="HWC")  # 将图像数据添加到TensorBoard可视化中，需指定数据格式为HWC（Height-Width-Channel）
        for i in range(100):
            writer.add_scalar("x=y", i, i)  # 添加标量值到TensorBoard可视化中
        writer.close()  # 关闭SummaryWriter对象
其中，SummaryWriter是一个用于将TensorFlow的训练日志写入磁盘以供TensorBoard使用的实用程序。
 #### 添加图像
 要将图像数据添加到TensorBoard可视化中，可以使用SummaryWriter的add_image()方法。例如，在上述代码中，我们使用如下代码将图像数据添加到TensorBoard可视化中：


    image_path = 'Dataset/train/ants_image/0013035.jpg'  # 图像文件路径
    image = Image.open(image_path)  # 打开图像文件
    image = np.array(image)  # 将图像文件转换成numpy数组
    writer.add_image('img', image, global_step=0,
                     dataformats="HWC")  # 将图像数据添加到TensorBoard可视化中，需指定数据格式为HWC（Height-Width-Channel）
可以看到，我们首先使用PIL库中的Image.open()方法打开图像文件，然后使用numpy库中的np.array()方法将图像文件转换成numpy数组。接下来，我们使用SummaryWriter对象的add_image()方法将图像数据添加到TensorBoard可视化中。需要注意的是，我们需要通过dataformats参数指定图像数据的格式，这里我们使用了HWC（Height-Width-Channel）格式。
 #### 添加标量值
 要将标量值添加到TensorBoard可视化中，可以使用SummaryWriter的add_scalar()方法。例如，在上述代码中，我们使用如下代码将100个标量值添加到TensorBoard可视化中：
        
    for i in range(100):
            writer.add_scalar("x=y", i, i)  # 添加标量值到TensorBoard可视化中
可以看到，我们使用for循环遍历100次，每一次向TensorBoard可视化中添加一个标量值。在add_scalar()方法中，第一个参数表示标量值的名称，在本例中我们使用"x=y"作为名称；第二个参数表示标量值的值，即在本例中为i；第三个参数表示标量值的步骤数，即在本例中为i。
 最后，我们使用writer.close()方法关闭SummaryWriter对象。



