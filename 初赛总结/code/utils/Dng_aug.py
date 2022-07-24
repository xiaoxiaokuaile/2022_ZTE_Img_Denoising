# Learner: 王振强
# Learn Time: 2022/4/9 9:54
import cv2
import random
import rawpy
import numpy as np
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from config.config import *


# 翻转图像
def RondomFlip(image,label):
    img_temp = image
    label_temp = label
    degree = random.random()
    if degree <= 0.33:
        img_temp = cv2.flip(img_temp, 0)
        label_temp = cv2.flip(label_temp, 0)
    else:
        img_temp = cv2.flip(img_temp, 1)
        label_temp = cv2.flip(label_temp, 1)
    return img_temp,label_temp


# 随机旋转
def RandomRotate(image, label, angle):
    img_temp = image
    label_temp = label
    h = img_temp.shape[0]
    w = img_temp.shape[1]

    angle = random.random() * angle
    angle = angle if random.random() < 0.5 else - angle
    scale = random.random() * 0.4 + 0.9
    matRotate = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, scale)
    img_temp = cv2.warpAffine(img_temp, matRotate, (w, h))
    label_temp = cv2.warpAffine(label_temp, matRotate, (w, h),)
    return img_temp, label_temp


# cutout 数据增强
def Cutout(n_holes, length, image, label):
    img_temp = image
    label_temp = label

    h = img_temp.shape[0]
    w = img_temp.shape[1]

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - int(length * h) // 2, 0, h)
        y2 = np.clip(y + int(length * h) // 2, 0, h)
        x1 = np.clip(x - int(length * w) // 2, 0, w)
        x2 = np.clip(x + int(length * w) // 2, 0, w)
        # black_level
        img_temp[y1: y2, x1: x2] = 0
        label_temp[y1: y2, x1: x2] = 0

    return img_temp,label_temp


# 翻转图像
def RondomRatio(image, label):
    img_temp = image
    label_temp = label
    img_diff = img_temp - label_temp  # 获取噪声
    degree = random.random()
    # 缩放
    if degree <= 0.33:
        img_temp = label_temp + img_diff*random.uniform(0.5,0.75)
    # 扩增
    else:
        img_temp = label_temp + img_diff*random.uniform(1.5,2)

    return img_temp,label_temp


def warp(img,label):
    """
    warp
    """
    new_img = img
    new_label = label

    prob_flip = 0.5     # 随机裁剪
    prob_Rotate = 0     # 随机旋转
    prob_Cutout = 0     # Cutout概率
    prob_ratio = 0.5    # 噪声随机缩放

    # 随机裁剪
    if random.random() <= prob_flip:
        # 参数依次为 小方块个数 遮盖比例
        new_img,new_label = RondomFlip(new_img, new_label)
    # 随机旋转
    if random.random() <= prob_Rotate:
        # 参数依次为 小方块个数 遮盖比例
        new_img,new_label = RandomRotate(new_img, new_label, 10)
    # cutout
    if random.random() <= prob_Cutout:
        # 参数依次为 小方块个数 遮盖比例
        new_img,new_label = Cutout(1, 0.5, new_img, new_label)

    # 随机放缩噪声
    if random.random() <= prob_ratio:
        # 参数依次为 小方块个数 遮盖比例
        new_img,new_label = RondomRatio(new_img, new_label)

    return new_img,new_label


if __name__ == '__main__':
    # 遍历文件夹
    noise_path = r"F:\JS\2022中兴捧月\code\data\train/noisy/1_noise.dng"       # 带噪声图像
    label_path = r"F:\JS\2022中兴捧月\code\data\train/gt/1_gt.dng"  # 去除噪声图像
    # 读取带噪声图像
    noise_raw = rawpy.imread(noise_path)  # dng读取为raw格式
    noise_np = np.array(noise_raw.raw_image_visible).astype(np.float32) # raw 格式转化为数组形式 (3472, 4624)
    # print(noise_np.shape)
    # noise_rgb = noise_raw.postprocess(use_camera_wb=True, half_size=True)
    # 读取去除噪声图像
    label_raw = rawpy.imread(label_path)    # dng读取为raw格式
    label_np = np.array(label_raw.raw_image_visible).astype(np.float32)  # raw 格式转化为数组形式 (3472, 4624)
    # label_rgb = label_raw.postprocess(use_camera_wb=True, half_size=True) # 得到rgb图像 (4624, 3472, 3)  [0,255]

    # ---------- 显示原图 ----------
    # plt.figure()   # 显示原图
    # plt.subplot(1, 2, 1)
    # plt.imshow(noise_np)
    # plt.subplot(1, 2, 2)
    # plt.imshow(label_np)
    print(noise_np)
    print(label_np)

    warm_img,warm_label = warp(noise_np, label_np)
    print(warm_img)
    print(warm_label)
    # -------- 显示处理后的图 -------
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(warm_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(warm_label)
    # plt.show()





























