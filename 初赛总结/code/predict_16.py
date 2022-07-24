import os
import numpy as np
import rawpy
import torch
import glob
import cv2
import skimage.metrics
from utils.utils import normalization,inv_normalization,read_image,write_image,write_back_dng
from config.config import *
from model import Unet,UNet1Plus,UNet2Plus,UNet3Plus
from model import B0_UNet1,B3_UNet1
from model import B0_UNet2,B3_UNet2
from model import Restormer,DRUNet,MPRNet

"""
    切分16张图拼接预测
"""


def predict(net, input_path, output_path, black_level, white_level):
    # 获取文件夹中 .dng 图片列表
    dng_list = glob.glob(os.path.join(input_path, '*.bin'))
    # 遍历文件夹
    for dng_path in dng_list:
        print(dng_path)
        # 读取数据
        img_rd = np.frombuffer(open(dng_path, 'rb').read(), dtype='uint16').reshape(896, 1216)
        # 预处理输入图像 -> 拆分成4通道
        raw_data_expand_c, height, width = read_image(img_rd)
        # int16->float 归一化
        raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)
        # 转化为tensor格式
        raw_data_expand_c_normal = torch.from_numpy(np.transpose(raw_data_expand_c_normal.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()
        # inference  推理
        # ----------------------------------------------------------
        output = net(raw_data_expand_c_normal)
        result_data = output
        # ----------------------------------------------------------
        # 后处理
        result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
        # 将网络输出结果转化为 int16 类型
        result_data = inv_normalization(result_data, black_level, white_level)
        # 图像拼接为原图
        result_write_data = write_image(result_data, height, width)
        # 保存预测图像
        save_name = dng_path.split('.')[-2].split('\\')[-1] + ".bin"
        # 重新保存切片
        fout = open(os.path.join(output_path, save_name), 'wb')
        fout.write(result_write_data.tobytes())


# 将 16 张图像拼接为 (3472, 4624) 大小的图像并保存为 .dng格式
def img16_to_Img(src_bin_path, src_dng_path, save_dng_path):
    # 初始化 (3472, 4624) 大小的图像
    height = 3472
    width = 4624
    Img_out = np.zeros((height, width), dtype=np.uint16)
    # 16个点的中心坐标
    center = [[(434, 578), (434, 1734), (434, 2890), (434, 4046)],
              [(1302, 578), (1302, 1734), (1302, 2890), (1302, 4046)],
              [(2170, 578), (2170, 1734), (2170, 2890), (2170, 4046)],
              [(3038, 578), (3038, 1734), (3038, 2890), (3038, 4046)]]
    # 无overlap图
    clip_h = 434
    clip_w = 578
    # 切片图
    Img_h = 448
    Img_w = 608

    # 遍历测试集10张图片
    for i in range(10):
        pre_name = "noisy" + str(i)  # 获取前缀
        # 遍历 4 张图片(左上角, 左下角, 右上角, 右下角)
        for x_idx in range(4):
            for y_idx in range(4):
                clip_name = pre_name + "_" + str(x_idx) + "_" + str(y_idx) + ".bin"
                # 读取clip
                content = open(os.path.join(src_bin_path, clip_name), 'rb').read()
                clip = np.frombuffer(content, dtype='uint16').reshape(896, 1216)

                if (x_idx,y_idx) == (0,0):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[0:int(clip_h*2), 0:int(clip_w*2)]
                elif (x_idx,y_idx) == (0,1):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[0:int(clip_h*2), Img_w-clip_w:Img_w+clip_w]
                elif (x_idx,y_idx) == (0,2):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[0:int(clip_h*2), Img_w-clip_w:Img_w+clip_w]
                elif (x_idx,y_idx) == (0,3):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[0:int(clip_h*2), -int(clip_w*2):]
                elif (x_idx,y_idx) == (1,0):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[Img_h-clip_h:Img_h+clip_h, 0:int(clip_w*2)]
                elif (x_idx,y_idx) == (1,1):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[Img_h-clip_h:Img_h+clip_h, Img_w-clip_w:Img_w+clip_w]
                elif (x_idx,y_idx) == (1,2):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[Img_h-clip_h:Img_h+clip_h, Img_w-clip_w:Img_w+clip_w]
                elif (x_idx,y_idx) == (1,3):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[Img_h-clip_h:Img_h+clip_h, -int(clip_w*2):]
                elif (x_idx,y_idx) == (2,0):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[Img_h-clip_h:Img_h+clip_h, 0:int(clip_w*2)]
                elif (x_idx,y_idx) == (2,1):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[Img_h-clip_h:Img_h+clip_h, Img_w-clip_w:Img_w+clip_w]
                elif (x_idx,y_idx) == (2,2):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[Img_h-clip_h:Img_h+clip_h, Img_w-clip_w:Img_w+clip_w]
                elif (x_idx,y_idx) == (2,3):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[Img_h-clip_h:Img_h+clip_h, -int(clip_w*2):]
                elif (x_idx,y_idx) == (3,0):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[-int(clip_h*2):, 0:int(clip_w*2)]
                elif (x_idx,y_idx) == (3,1):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[-int(clip_h*2):, Img_w-clip_w:Img_w+clip_w]
                elif (x_idx,y_idx) == (3,2):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[-int(clip_h*2):, Img_w-clip_w:Img_w+clip_w]
                elif (x_idx,y_idx) == (3,3):
                    Img_out[center[x_idx][y_idx][0]-clip_h:center[x_idx][y_idx][0]+clip_h,
                            center[x_idx][y_idx][1]-clip_w:center[x_idx][y_idx][1]+clip_w] = \
                        clip[-int(clip_h*2):, -int(clip_w*2):]
                else:
                    print("ERROR!!!")
        # 将拼接的图像转化为dng格式
        src_path = os.path.join(src_dng_path, pre_name + ".dng")  # 参照的 .dng 图像地址
        dest_path = os.path.join(save_dng_path, "denoise" + str(i) + ".dng")
        write_back_dng(src_path, dest_path, Img_out)


if __name__ == '__main__':
    # -------------- 单模型预测 -------------
    model_pr = UNet1Plus()
    model_pr.load_model("./weights/UNet1Plus_best.pth")
    model_pr.eval()
    # 预测noise路径
    test_path = "./data/test/noise_overlap_16"
    # 预测gt路径
    test_output_path = "./data/test/pre_16"
    print("切片预测开始！")
    predict(model_pr, test_path, test_output_path, black_level, white_level)
    print("切片预测结束！")
    # ------------ 多模型融合预测 ------------

    # 将64张图像拼接为 (3472, 4624) 大小的图像
    src_bin_path = "./data/test/pre_16"
    src_dng_path = "./data/test/noise"
    save_dng_path = "./data/test/pre_gt"
    print("切片拼接开始！")
    img16_to_Img(src_bin_path, src_dng_path, save_dng_path)
    print("切片拼接结束！")

