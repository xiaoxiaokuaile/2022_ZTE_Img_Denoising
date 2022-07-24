# Learner: 王振强
# Learn Time: 2022/4/8 18:02
import csv
import rawpy
from matplotlib import pyplot as plt
import glob
import os
import cv2
import numpy as np

"""
    1.将.dng格式图片转化为jpg
    2.获取 black_level , white_level
    3.将 (3472, 4624) 的原始图像转化为 64 张 (434,578) 大小的 .bin 图像
    4.制作 bin 数据五折交叉表
"""


# 1.将.dng格式图片转化为jpg
def Dng_To_Jpg(src_path, save_path):
    # 获取文件夹中 .dng 图片列表
    dng_list = glob.glob(os.path.join(src_path, '*.dng'))
    # 遍历文件夹
    for dng_path in dng_list:
        read_dng = rawpy.imread(dng_path)
        """
            use_camera_wb 是否执行自动白平衡，如果不执行白平衡，一般图像会偏色
            half_size 是否图像减半
            no_auto_bright 不自动调整亮度
            output_bps bit数据， 8或16
        """
        img = read_dng.postprocess(use_camera_wb=True)
        print(img.shape)
        # 获取图片名称
        img_name = dng_path.split('.')[-2].split('\\')[-1]
        cv2.imwrite(os.path.join(save_path, img_name + '.jpg'), img)


# 2.获取 black_level , white_level
def Dng_T(src_path):
    # 获取文件夹中 .dng 图片列表
    dng_list = glob.glob(os.path.join(src_path, '*.dng'))
    # 遍历文件夹
    black_level = 10000
    white_level = 0
    for d in dng_list:
        read_dng = rawpy.imread(d).raw_image_visible  # (3472, 4624)
        # print(read_dng.shape)
        min_lavel = np.min(read_dng)
        max_lavel = np.max(read_dng)
        print(min_lavel, max_lavel)
        if min_lavel < black_level:
            black_level = min_lavel
        if max_lavel > white_level:
            white_level = max_lavel
    print("black_level:",black_level,"white_level",white_level)


# 3.将 (3472, 4624) 的原始图像转化为 320 张 (256,256) 大小的 .bin 图像
# 每个角 4×5=20 张,
def Img_to_320(src_path,save_path):
    # 获取文件夹中 .dng 图片列表
    global clip_img
    dng_list = glob.glob(os.path.join(src_path, '*.dng'))
    # 遍历文件夹
    for dng_path in dng_list:
        dng_name = dng_path.split('.')[-2].split('\\')[-1]
        print(dng_name)
        read_dng = rawpy.imread(dng_path).raw_image_visible  # (3472, 4624)
        height = 3472
        width = 4624
        # 遍历4个角
        idx_num = 0
        # 块大小
        pixel = 256
        for point in range(4):
            for h_idx in range(8):
                for w_idx in range(10):
                    idx_num = idx_num + 1
                    clip_name = dng_name + "_" + str(idx_num)
                    fout = open(os.path.join(save_path, clip_name + ".bin"), 'wb')
                    # 左上角
                    if point == 0:
                        clip_img = read_dng[pixel*h_idx:pixel*(h_idx+1), pixel*w_idx:pixel*(w_idx+1)]
                        fout.write(clip_img.tobytes())
                    # 左下角
                    elif point == 1:
                        clip_img = read_dng[height-pixel*(h_idx+1):height-pixel*h_idx, pixel*w_idx:pixel*(w_idx+1)]
                        fout.write(clip_img.tobytes())
                    # 右上角
                    elif point == 2:
                        clip_img = read_dng[pixel*h_idx:pixel*(h_idx+1), width-pixel*(w_idx+1):width-pixel*w_idx]
                        fout.write(clip_img.tobytes())
                    # 右下角
                    elif point == 3:
                        clip_img = read_dng[height-pixel*(h_idx+1):height-pixel*h_idx, width-pixel*(w_idx+1):width-pixel*w_idx]
                        fout.write(clip_img.tobytes())
                    else:
                        print("ERROR!!!")


# 4.将 (3472, 4624) 的原始图像转化为 4 张 (1792,2368) 大小的 .bin 图像
def Img_to_4(src_path,save_path):
    # 获取文件夹中 .dng 图片列表
    global clip_img
    dng_list = glob.glob(os.path.join(src_path, '*.dng'))
    # 遍历文件夹
    for dng_path in dng_list:
        dng_name = dng_path.split('.')[-2].split('\\')[-1]
        print(dng_name)
        read_dng = rawpy.imread(dng_path).raw_image_visible  # (3472, 4624)
        # plt.figure()
        # plt.imshow(read_dng)
        height = 3472
        width = 4624
        # 遍历4个角
        # plt.figure()  # 显示原图
        for point in range(1,5):
                clip_name = dng_name + "_" + str(point)
                fout = open(os.path.join(save_path, clip_name + ".bin"), 'wb')
                # 左上角
                if point == 1:
                    # clip_img = read_dng[0:height//2, 0:width//2]  # 平均切
                    clip_img = read_dng[0:1792, 0:2368]  # overlap切 (1920,2560)
                    fout.write(clip_img.tobytes())
                    # plt.subplot(2, 2, 1)
                    # plt.imshow(clip_img)
                # 左下角
                elif point == 2:
                    # clip_img = read_dng[height//2:, 0:width//2]
                    clip_img = read_dng[-1792:, 0:2368]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(2, 2, 3)
                    # plt.imshow(clip_img)
                # 右上角
                elif point == 3:
                    # clip_img = read_dng[0:height//2, width//2:]
                    clip_img = read_dng[0:1792, -2368:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(2, 2, 2)
                    # plt.imshow(clip_img)
                # 右下角
                elif point == 4:
                    # clip_img = read_dng[height//2:, width//2:]
                    clip_img = read_dng[-1792:, -2368:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(2, 2, 4)
                    # plt.imshow(clip_img)
                else:
                    print("ERROR!!!")
        # plt.show()
        # break


# 4.将 (3472, 4624) 的原始图像转化为 16 张 (896, 1216) 大小的 .bin 图像
def Img_to_16(src_path,save_path):
    # 获取文件夹中 .dng 图片列表
    global clip_img
    dng_list = glob.glob(os.path.join(src_path, '*.dng'))
    # 遍历文件夹
    for dng_path in dng_list:
        dng_name = dng_path.split('.')[-2].split('\\')[-1]
        print(dng_name)
        read_dng = rawpy.imread(dng_path).raw_image_visible  # (3472, 4624)
        # plt.figure()
        # plt.imshow(read_dng)
        # 16个点的中心坐标
        center = [[(434,578), (434,1734), (434,2890), (434,4046) ],
                  [(1302,578),(1302,1734),(1302,2890),(1302,4046)],
                  [(2170,578),(2170,1734),(2170,2890),(2170,4046)],
                  [(3038,578),(3038,1734),(3038,2890),(3038,4046)]]
        # 切片大小
        clip_h = 896
        clip_w = 1216
        # 遍历4个角
        # plt.figure()  # 显示原图
        for x_idx in range(4):
            for y_idx in range(4):
                clip_name = dng_name + "_" + str(x_idx) + "_" + str(y_idx)
                fout = open(os.path.join(save_path, clip_name + ".bin"), 'wb')

                if (x_idx,y_idx) == (0,0):
                    clip_img = read_dng[0:clip_h, 0:clip_w]  # overlap切 (1920,2560)
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 1)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (0,1):
                    clip_img = read_dng[0:clip_h:, center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 2)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (0,2):
                    clip_img = read_dng[0:clip_h, center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 3)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (0,3):
                    clip_img = read_dng[0:clip_h:, -clip_w:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 4)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (1,0):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), 0:clip_w]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 5)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (1,1):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 6)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (1,2):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 7)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (1,3):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), -clip_w:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 8)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (2,0):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), 0:clip_w]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 9)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (2,1):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 10)
                    # plt.imshow(clip_img)
                elif (x_idx, y_idx) == (2, 2):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 11)
                    # plt.imshow(clip_img)
                elif (x_idx, y_idx) == (2, 3):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), -clip_w:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 12)
                    # plt.imshow(clip_img)
                elif (x_idx, y_idx) == (3, 0):
                    clip_img = read_dng[-clip_h:, 0:clip_w]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 13)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (3, 1):
                    clip_img = read_dng[-clip_h:, center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 14)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (3,2):
                    clip_img = read_dng[-clip_h:, center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 15)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (3,3):
                    clip_img = read_dng[-clip_h:, -clip_w:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 4, 16)
                    # plt.imshow(clip_img)
                else:
                    print("ERROR!!!")
        # plt.show()
        # break


def Img_to_20(src_path,save_path):
    # 获取文件夹中 .dng 图片列表
    global clip_img
    dng_list = glob.glob(os.path.join(src_path, '*.dng'))
    # 遍历文件夹
    for dng_path in dng_list:
        dng_name = dng_path.split('.')[-2].split('\\')[-1]
        print(dng_name)
        read_dng = rawpy.imread(dng_path).raw_image_visible  # (3472, 4624)
        # plt.figure()
        # plt.imshow(read_dng)
        # 32个点的中心坐标
        center = [[(434,462), (434,1386), (434,2310), (434,3234), (434,4160)],
                  [(1302,462),(1302,1386),(1302,2310),(1302,3234),(1302,4160)],
                  [(2170,462),(2170,1386),(2170,2310),(2170,3234),(2170,4160)],
                  [(3038,462),(3038,1386),(3038,2310),(3038,3234),(3038,4160)]]
        # 切片大小
        clip_h = 960
        clip_w = 960
        # 遍历4个角
        # plt.figure()  # 显示原图
        for x_idx in range(4):
            for y_idx in range(5):
                clip_name = dng_name + "_" + str(x_idx) + "_" + str(y_idx)
                fout = open(os.path.join(save_path, clip_name + ".bin"), 'wb')

                if (x_idx,y_idx) == (0,0):
                    clip_img = read_dng[0:clip_h, 0:clip_w]  # overlap切 (1920,2560)
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 1)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (0,1):
                    clip_img = read_dng[0:clip_h, center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 2)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (0,2):
                    clip_img = read_dng[0:clip_h, center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 3)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (0,3):
                    clip_img = read_dng[0:clip_h, center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 4)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (0,4):
                    clip_img = read_dng[0:clip_h, -clip_w:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 5)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (1,0):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), 0:clip_w]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 6)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (1,1):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 7)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (1,2):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 8)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (1,3):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 9)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (1,4):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), -clip_w:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 10)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (2,0):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), 0:clip_w]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 11)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (2,1):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 12)
                    # plt.imshow(clip_img)
                elif (x_idx, y_idx) == (2, 2):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 13)
                    # plt.imshow(clip_img)
                elif (x_idx, y_idx) == (2, 3):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 14)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (2, 4):
                    clip_img = read_dng[center[x_idx][y_idx][0]-int(clip_h/2):center[x_idx][y_idx][0]+int(clip_h/2), -clip_w:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 15)
                    # plt.imshow(clip_img)
                elif (x_idx, y_idx) == (3, 0):
                    clip_img = read_dng[-clip_h:, 0:clip_w]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 16)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (3, 1):
                    clip_img = read_dng[-clip_h:, center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 17)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (3, 2):
                    clip_img = read_dng[-clip_h:, center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 18)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (3,3):
                    clip_img = read_dng[-clip_h:,center[x_idx][y_idx][1]-int(clip_w/2):center[x_idx][y_idx][1]+int(clip_w/2)]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 19)
                    # plt.imshow(clip_img)
                elif (x_idx,y_idx) == (3,4):
                    clip_img = read_dng[-clip_h:, -clip_w:]
                    fout.write(clip_img.tobytes())
                    # plt.subplot(4, 5, 20)
                    # plt.imshow(clip_img)
                else:
                    print("ERROR!!!")
        # plt.show()
        # break


# 4.制作 bin 数据五折交叉表
def make_bin_kfold(save_csv):
    # 打开训练集CSV文档
    f = open(save_csv, "w", encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["train", "label", "fold"])
    # 先遍历获取csv文档
    for idx in range(100):
        fold = idx
        # 遍历 切片
        for idx_clip in range(80):
                train_name = str(idx) + "_noise_" + str(idx_clip+1) +".bin"
                label_name = str(idx) + "_gt_" + str(idx_clip+1) +".bin"
                csv_writer.writerow([train_name, label_name, fold])
    f.close()

    # # 五折交叉标记
    # train = pd.read_csv(save_csv)
    # N_FOLDS = 5
    # strat_kfold = KFold(n_splits=N_FOLDS, random_state=2022, shuffle=True)
    # for fold_id, (train_index, val_index) in enumerate(strat_kfold.split(train.index)):
    #     # 分割训练集验证集
    #     train.iloc[val_index, -1] = fold_id
    #     train['fold'] = train['fold'].astype('int')
    #     train.to_csv('./train_fold.csv', index=None)


# 图像探索
def Img_show():
    noise_path = "./test/noise/noisy0.dng"       # 带噪声图像
    # 读取带噪声图像
    noise_raw = rawpy.imread(noise_path)  # dng读取为raw格式
    noise_np = noise_raw.raw_image_visible  # raw 格式转化为数组形式
    noise_rgb = noise_raw.postprocess(use_camera_wb=True, half_size=True)  # raw 转化为 jpg 格式

    height = noise_np.shape[0]  # 3472
    width = noise_np.shape[1]   # 4624

    # 图像切分为4张
    img_1 = noise_np[0:height:2, 0:width:2]
    img_2 = noise_np[0:height:2, 1:width:2]
    img_3 = noise_np[1:height:2, 0:width:2]
    img_4 = noise_np[1:height:2, 1:width:2]

    # 另一种方法 (4,1736,2312)
    x = noise_np.reshape((height // 2, 2, width // 2, 2)).transpose((1, 3, 0, 2)).reshape((4, height // 2, width // 2))
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x_re = x.reshape((2, 2, height // 2, width // 2)).transpose((2, 0, 3, 1)).reshape((height, width))

    # ---------- 显示原图 ----------
    plt.figure()
    plt.imshow(noise_rgb)
    plt.figure()   # 显示原图
    plt.subplot(2, 2, 1)
    plt.imshow(x1)
    plt.subplot(2, 2, 2)
    plt.imshow(x2)
    plt.subplot(2, 2, 3)
    plt.imshow(x3)
    plt.subplot(2, 2, 4)
    plt.imshow(x4)
    plt.show()


if __name__ == '__main__':
    # # 图像格式转换
    # src_path = "../result/result/data"
    # save_path = "./jpg/test_gt/"
    # Dng_To_Jpg(src_path, save_path)

    # # 2.获取 black_level , white_level
    # src_path = "./train/ground truth"
    # save_path = "./train/noisy"
    # test_path = "./test"
    # Dng_T(test_path)

    # 3.将 (3472, 4624) 的原始图像转化为 320 张 (256,256) 大小的图像
    # src_path = r"F:\JS\2022中兴捧月\data\gt"
    # save_path = "./train/gt_256"
    # Img_to_320(src_path, save_path)

    # 3.将 (3472, 4624) 的原始图像转化为 8 张 (512,512) 大小的图像
    # src_path = r"./test/noise"
    # save_path = "./test/noise_overlap_16"
    # Img_to_16(src_path, save_path)

    # 3.将 (3472, 4624) 的原始图像转化为 32 张
    src_path = r"./test/noise"
    save_path = "./test/noise_overlap_20"
    Img_to_20(src_path, save_path)


    # # 4.制作 bin 数据五折交叉表
    # save_csv = "./train_512_fold.csv"
    # make_bin_kfold(save_csv)

    # 探索图像
    # Img_show()














