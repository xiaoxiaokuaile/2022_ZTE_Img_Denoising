import os
import numpy as np
import rawpy
import torch
import glob
import cv2
from config.config import *
import skimage.metrics
from utils.utils import normalization,inv_normalization,read_image,write_image,write_back_dng
from model import Unet,UNet1Plus,UNet2Plus,UNet3Plus
from model import B0_UNet1,B3_UNet1,B0_UNet2,B3_UNet2


"""
    原图直接预测
"""


def predict(net, input_path, output_path, black_level, white_level):
    # 获取文件夹中 .dng 图片列表
    dng_list = glob.glob(os.path.join(input_path, '*.dng'))
    # 设备参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 遍历文件夹
    for dng_path in dng_list:
        # 读取数据
        img_rd = rawpy.imread(dng_path).raw_image_visible  # 读取dng格式
        # 预处理输入图像 -> 拆分成4通道
        raw_data_expand_c, height, width = read_image(img_rd)
        # int16->float 归一化
        raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)
        # 转化为tensor格式
        raw_data_expand_c_normal = torch.from_numpy(np.transpose(raw_data_expand_c_normal.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()
        # inference  推理
        result_data = net(raw_data_expand_c_normal.to(device))  # torch.Size([1, 4, 1736, 2312])
        # 后处理
        result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)  # [1,1736,2312,4]
        # 将网络输出结果转化为 int16 类型
        result_data = inv_normalization(result_data, black_level, white_level)
        # 图像拼接为原图
        result_write_data = write_image(result_data, height, width) # (3472, 4624)
        # 保存预测图像
        idx = dng_path.split(".")[-2].split("/")[-1][5:]
        dest_path = os.path.join(output_path, "denoise" + str(idx) + ".dng")
        write_back_dng(dng_path, dest_path, result_write_data)


if __name__ == '__main__':
    # -------------- 单模型预测 -------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pr = UNet1Plus().to(device)
    model_pr.load_model("./weights/UNet1Plus_best.pth")
    model_pr.eval()
    # 预测noise路径
    test_path = "./data/test/noise"
    # 预测gt路径
    test_output_path = "./data/test/pre_gt"
    print("预测开始！")
    predict(model_pr, test_path, test_output_path, black_level, white_level)
    print("预测结束！")



