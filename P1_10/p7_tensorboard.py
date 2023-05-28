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
# 注释：
# SummaryWriter是一个用于将TensorFlow的训练日志写入磁盘以供TensorBoard使用的实用程序。
# Image是一个Python库，可以用来打开、操作和保存多种图像文件格式。
# numpy是一个Python第三方库，主要用于数组计算，提供了多种高效的数组操作方法。
# np.array()是numpy库中的函数，可以将Python中的列表或元组等对象转换成numpy数组。
# add_image()是SummaryWriter对象中的方法，用于将图像数据添加到TensorBoard可视化中。
# add_scalar()是SummaryWriter对象中的方法，用于将标量数值添加到TensorBoard可视化中。
