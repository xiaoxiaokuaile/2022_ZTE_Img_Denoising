# Learner: 王振强
# Learn Time: 2022/7/20 9:48
import h5py
import os
import csv
import time
import cv2
import numpy as np
import rawpy
import glob
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import random


# 归一化
def normalization(input_data, black_level, white_level):
    output_data = np.maximum(input_data.astype(float) - black_level, 0) / (white_level - black_level)
    return output_data


# 反归一化
def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


# 读取并显示 .dng 格式数据
def read_show_dng(black_level, white_level):
    noise = './data/noise/'
    gt = './data/gt/'
    # 保存需求数据
    path = glob.glob(gt + '*.dng')
    # 遍历数据集
    for idx in range(len(path)):
        bin_noise_name = str(idx) + '_noise.dng'
        bin_gt_name = str(idx) + '.dng'
        noise_path = os.path.join(noise, bin_noise_name)
        gt_path = os.path.join(gt, bin_gt_name)
        # 读取 noise 图
        noise_rd = np.array(rawpy.imread(noise_path).raw_image_visible).astype(np.float64)  # (3472, 4624)
        h = noise_rd.shape[0]
        w = noise_rd.shape[1]
        # 读取 gt 图
        gt_rd = np.array(rawpy.imread(gt_path).raw_image_visible).astype(np.float64)  # (3472, 4624)
        # 归一化
        gt_rd_normal = normalization(gt_rd, black_level, white_level)

        print("开始去噪: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        # gt_rd = cv2.GaussianBlur(noise_rd, (7,7), 0) # 高斯滤波
        # gt_rd = cv2.blur(noise_rd, (5,5), 0)  # 均值滤波
        # kernel = np.array([[1, -2, 1],
        #                    [-2, 4, -2],
        #                    [1, -2, 1]])
        # Z_diff = cv2.filter2D(noise_rd_normal, -1, kernel)
        denoise_rd = signal.medfilt(noise_rd, (3,3))  # 中值滤波
        denoise_rd_normal = normalization(denoise_rd, black_level, white_level)
        # 归一化
        noise_rd_normal = normalization(noise_rd, black_level, white_level)

        # 3472, 4624
        Y = np.linspace(1,3472,3472)
        X = np.linspace(1,4624,4624)
        xx, yy = np.meshgrid(X, Y)  # 网格化坐标
        Y1 = np.linspace(1,3472,3472)
        X1 = np.linspace(1,4624,4624)
        xx1, yy1 = np.meshgrid(X1, Y1)  # 网格化坐标
        Z_noise = noise_rd_normal
        Z_gt = gt_rd_normal
        Z_diff = noise_rd_normal - gt_rd_normal
        Z_denoise_diff = noise_rd_normal - denoise_rd_normal

        # 绘制 gt noise diff 图
        fig = plt.figure()
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot_surface(xx, yy, Z_noise, cmap='rainbow')
        plt.title(u'noise')
        ax2 = fig.add_subplot(223, projection='3d')
        ax2.plot_surface(xx1, yy1, Z_gt, cmap='rainbow')
        plt.title(u'gt')
        ax3 = fig.add_subplot(222, projection='3d')
        ax3.plot_surface(xx1, yy1, denoise_rd_normal, cmap='rainbow')
        plt.title(u'denoise normal')
        ax3 = fig.add_subplot(224, projection='3d')
        ax3.plot_surface(xx1, yy1, Z_denoise_diff, cmap='rainbow')
        plt.title(u'denoise-diff')
        plt.show()

        # 原图噪声分布绘制
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        Z_1D_diff = Z_diff.astype(np.float32).reshape(3472*4624)  # 归一化之后转化为float类型
        ax1.hist(Z_1D_diff, bins=256)
        plt.title("noise diff show")
        # save_name = "img_" + str(idx)+ '.jpg'
        # plt.savefig(os.path.join("./data_show/train_noise_hist", save_name), dpi=300)
        # 去噪噪声分布
        ax2 = fig.add_subplot(122)
        Z_2D_diff = Z_denoise_diff.astype(np.float32).reshape(3472*4624)  # 归一化之后转化为float类型
        ax2.hist(Z_2D_diff, bins=256)
        plt.title("denoise diff show")
        # save_name = "img_" + str(idx)+ '.jpg'
        # plt.savefig(os.path.join("./data_show/train_noise_hist", save_name), dpi=300)
        plt.show()

        # 切分图像
        pixel = 446
        point = []
        for x in range(h // pixel):
            for y in range(w // pixel):
                point.append([x, y])
        # 随机打乱顺序
        lt2 = [i for i in range(len(point))]

        a_out_list = []
        b_out_list = []
        # 迭代次数
        epochs = 1
        print("开始计算a,b ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        for i in range(epochs):
            random.shuffle(lt2)
            # 随机抽取两张256×256的切片
            h_idx1 = point[lt2[0]][0]
            w_idx1 = point[lt2[0]][1]
            clip1_noise = noise_rd_normal[pixel * h_idx1:pixel * (h_idx1 + 1), pixel * w_idx1:pixel * (w_idx1 + 1)]
            clip1_denoise_diff = Z_denoise_diff[pixel * h_idx1:pixel * (h_idx1 + 1), pixel * w_idx1:pixel * (w_idx1 + 1)]
            clip1_diff = Z_diff[pixel * h_idx1:pixel * (h_idx1 + 1), pixel * w_idx1:pixel * (w_idx1 + 1)]

            h_idx2 = point[lt2[1]][0]
            w_idx2 = point[lt2[1]][1]
            clip2_noise = noise_rd_normal[pixel * h_idx2:pixel * (h_idx2 + 1), pixel * w_idx2:pixel * (w_idx2 + 1)]
            clip2_denoise_diff = Z_denoise_diff[pixel * h_idx2:pixel * (h_idx2 + 1), pixel * w_idx2:pixel * (w_idx2 + 1)]
            clip2_diff = Z_diff[pixel * h_idx2:pixel * (h_idx2 + 1), pixel * w_idx2:pixel * (w_idx2 + 1)]

            # 切片图1噪声分布绘制
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            Z_1D_clip_img11 = clip1_diff.astype(np.float32).reshape(pixel*pixel)  # 归一化之后转化为float类型
            ax1.hist(Z_1D_clip_img11, bins=256)
            plt.title("clip1 diff show")
            ax2 = fig.add_subplot(122)
            Z_1D_clip_img12 = clip1_denoise_diff.astype(np.float32).reshape(pixel*pixel)  # 归一化之后转化为float类型
            ax2.hist(Z_1D_clip_img12, bins=256)
            plt.title("clip1 denoise show")
            plt.show()

            # 切片图2噪声分布绘制
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            Z_1D_clip_img21 = clip2_diff.astype(np.float32).reshape(pixel*pixel)  # 归一化之后转化为float类型
            ax1.hist(Z_1D_clip_img21, bins=256)
            plt.title("clip2 diff show")
            ax2 = fig.add_subplot(122)
            Z_1D_clip_img22 = clip2_denoise_diff.astype(np.float32).reshape(pixel*pixel)  # 归一化之后转化为float类型
            ax2.hist(Z_1D_clip_img22, bins=256)
            plt.title("clip2 denoise show")
            plt.show()

            # print("索引", "gt图均值", "noise图均值", "噪声均值", "噪声0.5分位数", "标准差",
            #       "-σ分位数", "+σ分位数", "-2σ分位数", "+2σ分位数", "-3σ分位数", "+3σ分位数")
            # print(idx,
            #       round(float(np.mean(gt_rd_normal)),6),
            #       round(float(np.mean(noise_rd_normal)),6),
            #       round(float(np.mean(Z_diff)),6),
            #       round(float(np.percentile(Z_diff, 50)), 6),
            #       round(float(np.std(Z_diff)),6),
            #       round(float(np.percentile(Z_diff,15.8655255)),6),
            #       round(float(np.percentile(Z_diff,84.1344745)),6),
            #       round(float(np.percentile(Z_diff, 2.275013)), 6),
            #       round(float(np.percentile(Z_diff, 97.724987)), 6),
            #       round(float(np.percentile(Z_diff, 0.13499)), 6),
            #       round(float(np.percentile(Z_diff, 99.86501)), 6),
            #       )
            # print("索引", "gt图clip1均值", "noise图clip1均值", "clip1噪声均值", "clip10.5分位数", "clip1标准差",
            #       "-σ分位数", "+σ分位数", "-2σ分位数", "+2σ分位数")
            # print(idx,
            #       # round(float(np.mean(clip1_gt)),6),
            #       round(float(np.mean(clip1_noise)),6),
            #       round(float(np.mean(clip1_diff)),6),
            #       round(float(np.percentile(clip1_diff, 50)), 6),
            #       round(float(np.std(clip1_diff)),6),
            #       round(float(np.percentile(clip1_diff,15.8655255)),6),
            #       round(float(np.percentile(clip1_diff,84.1344745)),6),
            #       round(float(np.percentile(clip1_diff, 2.275013)), 6),
            #       round(float(np.percentile(clip1_diff, 97.724987)), 6),
            #       # round(float(np.percentile(clip1_diff, 0.13499)), 6),
            #       # round(float(np.percentile(clip1_diff, 99.86501)), 6),
            #       )
            #
            # print("索引", "gt图clip2均值", "noise图clip2均值", "clip2噪声均值", "clip2 0.5分位数", "clip2标准差",
            #       "-σ分位数", "+σ分位数", "-2σ分位数", "+2σ分位数")
            # print(idx,
            #       # round(float(np.mean(clip2_gt)),6),
            #       round(float(np.mean(clip2_noise)),6),
            #       round(float(np.mean(clip2_diff)),6),
            #       round(float(np.percentile(clip2_diff, 50)), 6),
            #       round(float(np.std(clip2_diff)),6),
            #       round(float(np.percentile(clip2_diff,15.8655255)),6),
            #       round(float(np.percentile(clip2_diff,84.1344745)),6),
            #       round(float(np.percentile(clip2_diff, 2.275013)), 6),
            #       round(float(np.percentile(clip2_diff, 97.724987)), 6),
            #       # round(float(np.percentile(clip2_diff, 0.13499)), 6),
            #       # round(float(np.percentile(clip2_diff, 99.86501)), 6),
            #       )

            # fig = plt.figure()
            # ax1 = fig.add_subplot(131, projection='3d')
            # ax1.plot_surface(xx, yy, Z_noise,cmap='rainbow')
            # plt.title(u'noise')
            # ax2 = fig.add_subplot(132, projection='3d')
            # ax2.plot_surface(xx1, yy1, Z_gt,cmap='rainbow')
            # plt.title(u'gt')
            # ax3 = fig.add_subplot(133, projection='3d')
            # ax3.plot_surface(xx1, yy1, Z_diff,cmap='rainbow')
            # plt.title(u'noise-gt')
            # save_name = "img_" + str(idx)+ '.jpg'
            # plt.savefig(os.path.join("./data_show/train_noise_show", save_name), dpi=300)
            # plt.show()

            gt1 = float(np.mean(clip1_noise))
            sigma1 = float(np.percentile(clip1_diff, 84.1344745))
            gt2 = float(np.mean(clip2_noise))
            sigma2 = float(np.percentile(clip2_diff, 84.1344745))

            a_out = (sigma1 ** 2 - sigma2 ** 2) / (gt1 - gt2)
            b_out = sigma1 ** 2 - a_out * gt1

            a_out_list.append(a_out)
            b_out_list.append(b_out)

        a_out_list.sort()
        b_out_list.sort()
        # print(a_out_list)
        # print(b_out_list)

        start_idx = int(0.2*epochs)
        end_idx = int(0.8*epochs)
        # a_last = np.clip(np.mean(a_out_list[start_idx:end_idx]), 0.0001, 0.01)
        a_last = np.clip(float(np.percentile(a_out_list, 50)), 0.0001, 0.01)
        # b_last = np.clip(np.mean(b_out_list[start_idx:end_idx]), 0.000001, 0.001)
        b_last = np.clip(float(np.percentile(b_out_list, 50)), 0.000001, 0.001)
        evm_tmp = 0.5 * np.min([np.abs(0.001 - a_last) / 0.001, 1]) + 0.5 * np.min([np.abs(0.0001 - b_last) / 0.0001, 1])
        score_tmp = 100 - 100 * evm_tmp

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), end='')
        print("a预测: ", a_last, "b预测: ", b_last, "得分: ", score_tmp)

        # break


if __name__ == '__main__':
    black_level = 1024
    white_level = 16383
    read_show_dng(black_level, white_level)






