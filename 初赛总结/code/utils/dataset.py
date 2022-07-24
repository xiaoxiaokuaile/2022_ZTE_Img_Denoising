# Learner: 王振强
# Learn Time: 2022/4/8 17:54
import os
import numpy as np
from torch.utils import data
from .utils import normalization,read_image
from config.config import *
from .Dng_aug import warp


def img_norm_tensor(img_rd, label_rd):
    # img 归一化
    img_p, h, w = read_image(img_rd)  # (1736, 2312, 4)
    train_normal = normalization(img_p, black_level, white_level)
    train_tensor = np.transpose(train_normal, (2, 0, 1)).astype(np.float32)  # torch.Size([4, 1736, 2312])
    # label 归一化
    label_p, h, w = read_image(label_rd)  # (1736, 2312, 4)
    label_normal = normalization(label_p, black_level, white_level)
    label_tensor = np.transpose(label_normal, (2, 0, 1)).astype(np.float32)  # torch.Size([4, 1736, 2312])

    return train_tensor, label_tensor


# 加载数据集
class Captcha(data.Dataset):
    # 输入已经读取的CSV表格, 图片地址, 是否测试集
    def __init__(self, csv, img_path, label_path, data_mode='train'):
        self.data = csv
        self.img_path = img_path      # img 文件夹地址
        self.label_path = label_path  # label 文件夹地址
        self.data_mode = data_mode    # 判定 train , val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ImgName, label = self.data.iloc[index, :]
        # --------------- 读取 img -----------------
        imgPath = os.path.join(self.img_path, ImgName)
        # img_rd = rawpy.imread(imgPath).raw_image_visible  # 读取dng格式
        img_rd = np.frombuffer(open(imgPath, 'rb').read(), dtype='uint16').reshape(height, width).astype(np.float32)
        # -------------- 读取 label ----------------
        labelPath = os.path.join(self.label_path, label)
        # label_rd = rawpy.imread(labelPath).raw_image_visible  # 读取dng格式
        label_rd = np.frombuffer(open(labelPath, 'rb').read(), dtype='uint16').reshape(height, width).astype(np.float32)
        # -----------------------------------------
        if self.data_mode == 'train':
            # 数据增强
            img_rd, label_rd = warp(img_rd, label_rd)
            # noise, label 图像归一化并转化为 tensor
            train_tensor,label_tensor = img_norm_tensor(img_rd, label_rd)
            label_int = label_rd.astype(np.float32)
            return train_tensor, label_tensor, label_int
        elif self.data_mode == 'val':
            # noise, label 图像归一化并转化为 tensor
            train_tensor,label_tensor = img_norm_tensor(img_rd, label_rd)
            label_int = label_rd.astype(np.float32)
            return train_tensor, label_tensor, label_int
































