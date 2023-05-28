# 导入必要的库
import cv2 as cv
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 指定图片路径
img_path = 'Dataset/train/ants_image/0013035.jpg'
# 打开图片并转成OpenCV格式和PIL格式
img = Image.open(img_path)
img_cv = cv.imread(img_path)
# 创建tensorboard写入器
writer = SummaryWriter('log')
# 定义图像转tensor的变换
tensor_trans = transforms.ToTensor()
# 将OpenCV格式和PIL格式的图片转成tensor并在tensorboard中添加
tensor_pic_cv = tensor_trans(img_cv)
tensor_pic = tensor_trans(img)
flip_trans = transforms.RandomHorizontalFlip(p=1)
flip_img = flip_trans(tensor_pic)
crop_trans = transforms.RandomCrop(size=(100, 100))
crop_img = crop_trans(tensor_pic)
resize_trans = transforms.Resize(size=(600, 600), antialias=True)
resize_img = resize_trans(img)
resize_img = tensor_trans(resize_img)
resize_trans2=transforms.Resize(600)
trans_compose=transforms.Compose([resize_trans2,tensor_trans])
resize_pic_2=trans_compose(img)
cet_crop_trans = transforms.CenterCrop(size=(50, 50))
cet_crop_img = cet_crop_trans(tensor_pic)
normalize_trans = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
normalize_img = normalize_trans(tensor_pic)
color_jitter_trans = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
color_jitter_pic = color_jitter_trans(tensor_pic)
writer.add_image('tensor', tensor_pic, 0)
writer.add_image('tensor_cv', tensor_pic_cv, 1)
writer.add_image('flip_pic', flip_img, 2)
writer.add_image('crop_pic', crop_img, 3)
writer.add_image('resize_pic', resize_img, 4)
writer.add_image('cet_crop_pic', cet_crop_img, 5)
writer.add_image('color_jitter_pic', color_jitter_pic, 6)
writer.add_image('normalize_pic', normalize_img, 7)
writer.add_image('resi_pic',resize_pic_2, 8)

# 关闭tensorboard写入器
writer.close()
