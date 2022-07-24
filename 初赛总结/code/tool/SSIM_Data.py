# Learner: 王振强
# Learn Time: 2022/5/3 13:25
# from scipy.io import loadmat
import h5py
import os
import csv
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import random


# 将SSIM中.mat数据集转化为(256,256)大小数据集, 四个角各切60张
def SSIM_TO_256(file_path,save_path,mode):
    black = 1024
    write = 16383
    file_list = os.listdir(file_path)
    for file in file_list:
        temp_path = os.path.join(file_path,file)
        data_list = os.listdir(temp_path)

        mat_file = ""
        if mode == "noise":
            mat_file = os.path.join(temp_path, data_list[2])
        elif mode == "gt":
            mat_file = os.path.join(temp_path, data_list[0])
        else:
            print("ERROR!!!")

        image_ = h5py.File(mat_file)
        img2_array = np.array(image_['x'])  # (5328, 3000)
        height,width = img2_array.shape
        num_h = int(height/512 + 1)
        num_w = int(width/512 + 1)

        # print(height, width, num_h, num_w)
        img2_array_int16 = (img2_array*(write - black) + black).astype(np.uint16)
        print('noise',np.min(img2_array_int16),np.max(img2_array_int16))

        # 遍历4个角
        idx_num = 0
        # 块大小
        pixel = 256
        for point in range(4):
            for h_idx in range(num_h):
                for w_idx in range(num_w):
                    idx_num = idx_num + 1
                    clip_name = file + "_" + str(idx_num)
                    fout = open(os.path.join(save_path, clip_name + ".bin"), 'wb')
                    # 左上角
                    if point == 0:
                        clip_img = img2_array_int16[pixel*h_idx:pixel*(h_idx+1), pixel*w_idx:pixel*(w_idx+1)]
                        fout.write(clip_img.tobytes())
                    # 左下角
                    elif point == 1:
                        clip_img = img2_array_int16[height-pixel*(h_idx+1):height-pixel*h_idx, pixel*w_idx:pixel*(w_idx+1)]
                        fout.write(clip_img.tobytes())
                    # 右上角
                    elif point == 2:
                        clip_img = img2_array_int16[pixel*h_idx:pixel*(h_idx+1), width-pixel*(w_idx+1):width-pixel*w_idx]
                        fout.write(clip_img.tobytes())
                    # 右下角
                    elif point == 3:
                        clip_img = img2_array_int16[height-pixel*(h_idx+1):height-pixel*h_idx, width-pixel*(w_idx+1):width-pixel*w_idx]
                        fout.write(clip_img.tobytes())
                    else:
                        print("ERROR!!!")


# 将SSIM中.mat数据集转化为(256,256)大小数据集, 四个角各切60张
def DND_TO_256(file_path,save_path,mode):
    black = 1024
    write = 16383
    file_list = os.listdir(file_path)
    for file in file_list:
        # 获取 .mat 文件路径
        mat_file = os.path.join(file_path, file)

        image_ = h5py.File(mat_file)
        print(image_.keys())
        img2_array = np.array(image_['Inoisy'])  # (5328, 3000)
        print(img2_array)
        height,width = img2_array.shape
        num_h = int(height/512 + 1)
        num_w = int(width/512 + 1)

        # print(height, width, num_h, num_w)
        img2_array_int16 = (img2_array*(write - black) + black).astype(np.uint16)
        print('noise',np.min(img2_array_int16),np.max(img2_array_int16))

        # # 遍历4个角
        # idx_num = 0
        # # 块大小
        # pixel = 256
        # for point in range(4):
        #     for h_idx in range(num_h):
        #         for w_idx in range(num_w):
        #             idx_num = idx_num + 1
        #             clip_name = file + "_" + str(idx_num)
        #             fout = open(os.path.join(save_path, clip_name + ".bin"), 'wb')
        #             # 左上角
        #             if point == 0:
        #                 clip_img = img2_array_int16[pixel*h_idx:pixel*(h_idx+1), pixel*w_idx:pixel*(w_idx+1)]
        #                 fout.write(clip_img.tobytes())
        #             # 左下角
        #             elif point == 1:
        #                 clip_img = img2_array_int16[height-pixel*(h_idx+1):height-pixel*h_idx, pixel*w_idx:pixel*(w_idx+1)]
        #                 fout.write(clip_img.tobytes())
        #             # 右上角
        #             elif point == 2:
        #                 clip_img = img2_array_int16[pixel*h_idx:pixel*(h_idx+1), width-pixel*(w_idx+1):width-pixel*w_idx]
        #                 fout.write(clip_img.tobytes())
        #             # 右下角
        #             elif point == 3:
        #                 clip_img = img2_array_int16[height-pixel*(h_idx+1):height-pixel*h_idx, width-pixel*(w_idx+1):width-pixel*w_idx]
        #                 fout.write(clip_img.tobytes())
        #             else:
        #                 print("ERROR!!!")


# 读取并显示 .bin 格式数据
def read_show_bin():
    noise = r'F:\JS\2022中兴捧月\code\data\train\noisy_256'
    gt = r'F:\JS\2022中兴捧月\code\data\train\gt_256'
    # 遍历100张图
    for idx in range(100):
        # 每张图的切片 320张
        for clip_idx in range(40,41):
            bin_noise_name = str(idx) + '_noise_' + str(clip_idx) + '.bin'
            bin_gt_name = str(idx) + '_gt_' + str(clip_idx) + '.bin'
            noise_path = os.path.join(noise, bin_noise_name)
            gt_path = os.path.join(gt, bin_gt_name)
            noise_rd = np.frombuffer(open(noise_path, 'rb').read(), dtype='uint16').reshape(256, 256).astype(np.float64)
            gt_rd = np.frombuffer(open(gt_path, 'rb').read(), dtype='uint16').reshape(256, 256).astype(np.float64)
            X = np.linspace(1,256,256)
            Y = np.linspace(1,256,256)
            xx, yy = np.meshgrid(X, Y)  # 网格化坐标
            X1 = np.linspace(1,256,256)
            Y1 = np.linspace(1,256,256)
            xx1, yy1 = np.meshgrid(X1, Y1)  # 网格化坐标
            Z_noise = noise_rd
            Z_gt = gt_rd
            Z_diff = noise_rd - gt_rd
            Z_abs = np.abs(Z_diff).astype(np.int32)
            print(idx,
                  round(float(np.mean(Z_diff)), 2),
                  round(float(np.mean(Z_abs)), 2),
                  round(float(np.min(Z_diff)), 2),
                  round(float(np.max(Z_diff)), 2),
                  round(float(np.std(Z_diff)), 2),
                  round(float(np.percentile(Z_diff, 1)), 2),
                  round(float(np.percentile(Z_diff, 99)), 2),
                  )

            fig = plt.figure()
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot_surface(xx, yy,Z_noise,cmap='rainbow')
            plt.title(u'noise')
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.plot_surface(xx1, yy1, Z_diff,cmap='rainbow')
            plt.title(u'gt')
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.plot_surface(xx1, yy1,Z_diff*random.uniform(0.5,2),cmap='rainbow')
            plt.title(u'noise-gt')
            plt.show()
            # save_name = "img_" + str(idx) + '_clip_' + str(clip_idx) + '.jpg'
            # plt.savefig(os.path.join("./train/img_show",save_name), dpi=300 )
            break



# 读取并显示 .dng 格式数据
def read_show_dng():
    noise = r'F:\JS\2022中兴捧月\code\data\train\noisy'
    gt = r'F:\JS\2022中兴捧月\code\data\train\gt'
    # 打开训练集CSV文档
    save_csv = r'F:\JS\2022中兴捧月\code\data\data_show/train.csv'
    f = open(save_csv, "w", encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["索引", "噪声均值","噪声绝对值均值", "最小值", "最大值", "方差", "0.01分位数", "0.99分位数"])
    # 遍历100张图
    for idx in range(100):
        bin_noise_name = str(idx) + '_noise.dng'
        bin_gt_name = str(idx) + '_gt.dng'
        noise_path = os.path.join(noise, bin_noise_name)
        gt_path = os.path.join(gt, bin_gt_name)
        noise_rd = np.array(rawpy.imread(noise_path).raw_image_visible).astype(np.float64)  # (3472, 4624)
        # print(noise_rd.shape)
        gt_rd = np.array(rawpy.imread(gt_path).raw_image_visible).astype(np.float64)  # (3472, 4624)
        # 3472, 4624
        Y = np.linspace(1,3472,3472)
        X = np.linspace(1,4624,4624)
        xx, yy = np.meshgrid(X, Y)  # 网格化坐标
        Y1 = np.linspace(1,3472,3472)
        X1 = np.linspace(1,4624,4624)
        xx1, yy1 = np.meshgrid(X1, Y1)  # 网格化坐标
        Z_noise = noise_rd
        Z_gt = gt_rd
        Z_diff = noise_rd - gt_rd
        Z_abs = np.abs(Z_diff).astype(np.int32)

        plt.figure()
        Z_1D_diff = Z_diff.astype(np.int32).reshape(3472*4624)
        plt.hist(Z_1D_diff, bins=128)
        plt.title("noise show")
        save_name = "img_" + str(idx)+ '.jpg'
        plt.savefig(os.path.join("./data_show/train_noise_hist", save_name), dpi=300)
        # plt.show()
        print(idx,
              round(float(np.mean(Z_diff)),2),
              round(float(np.mean(Z_abs)),2),
              round(float(np.min(Z_diff)),2),
              round(float(np.max(Z_diff)),2),
              round(float(np.std(Z_diff)),2),
              round(float(np.percentile(Z_diff,1)),2),
              round(float(np.percentile(Z_diff,99)),2),
              )
        csv_writer.writerow([
                             idx,
                             round(float(np.mean(Z_diff)),2),
                             round(float(np.mean(Z_abs)),2),
                             round(float(np.min(Z_diff)),2),
                             round(float(np.max(Z_diff)),2),
                             round(float(np.std(Z_diff)),2),
                             round(float(np.percentile(Z_diff,1)),2),
                             round(float(np.percentile(Z_diff,99)),2),
                             ])
        # fig = plt.figure()
        # ax1 = fig.add_subplot(131, projection='3d')
        # ax1.plot_surface(xx, yy,Z_noise,cmap='rainbow')
        # plt.title(u'noise')
        # ax2 = fig.add_subplot(132, projection='3d')
        # ax2.plot_surface(xx1, yy1,Z_gt,cmap='rainbow')
        # plt.title(u'gt')
        # ax3 = fig.add_subplot(133, projection='3d')
        # ax3.plot_surface(xx1, yy1,Z_diff,cmap='rainbow')
        # plt.title(u'noise-gt')
        # save_name = "img_" + str(idx)+ '.jpg'
        # plt.savefig(os.path.join("./train/img_show_test",save_name), dpi=300 )
        # plt.show()
        # break


# 4.制作 bin 数据五折交叉表
def make_bin_kfold(save_csv):
    # bin文档目录
    bin_file = r'F:\JS\2022中兴捧月\data\SIDD\SIDD_noise_256'
    bin_name_list = os.listdir(bin_file)
    # 打开训练集CSV文档
    f = open(save_csv, "w", encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["train", "label", "fold"])
    # 先遍历获取csv文档
    for bin_name in bin_name_list:
        csv_writer.writerow([bin_name, bin_name, 0])
    f.close()


if __name__ == '__main__':
    # file_path = r'F:\JS\2022中兴捧月\data\SIDD\Data'
    # save_path = r'F:\JS\2022中兴捧月\data\SIDD\SIDD_gt_256'
    # # SSIM_TO_256(file_path, save_path,"gt")

    file_path = r'F:\JS\2022中兴捧月\data\DND\images_raw'
    save_path = r'F:\JS\2022中兴捧月\data\DND\DND_noise_256'
    # DND_TO_256(file_path, save_path,"gt")

    # read_show_bin()
    read_show_dng()

    # save_csv = "SSIM_256_train.csv"
    # make_bin_kfold(save_csv)




