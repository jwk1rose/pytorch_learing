from typing import Tuple
# 导入必要的库
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, pic_dir: str, label_dir: str) -> None:
        """
        数据集类初始化方法
        :param pic_dir: 图片文件夹路径
        :param label_dir: 标签文件夹路径
        """
        self.label_dir = label_dir  # 标签文件夹路径
        self.pic_dir = pic_dir  # 图片文件夹路径
        self.image_list = os.listdir(pic_dir)  # 获取图片文件夹中所有图片文件名的列表
        self.label_list = os.listdir(label_dir)  # 获取标签文件夹中所有标签文件名的列表

    def __getitem__(self, item: int) -> Tuple[Image.Image, str]:
        """
        数据集类获取数据方法
        :param item: 索引值
        :return: 返回图片和标签
        """
        image_name = self.image_list[item]  # 获取当前index对应的图片文件名
        image_path = os.path.join(self.pic_dir, image_name)  # 拼接图片文件路径
        image = Image.open(image_path)  # 打开图片文件
        label_name = self.label_list[item]  # 获取当前index对应的标签文件名
        label_path = os.path.join(self.label_dir, label_name)  # 拼接标签文件路径
        with open(label_path, 'r') as f:
            label = f.read()  # 读取标签文件内容
        return image, label

    def __len__(self) -> int:
        """
        数据集类获取数据长度方法
        :return: 返回数据集长度
        """
        return self.image_list.__len__()


mydata = MyData(pic_dir="Dataset\\train\\ants_image", label_dir="Dataset\\train\\ants_label")  # 创建MyData类
