import os
import numpy as np
import rawpy
import glob
import argparse
import scipy.signal as signal
import time
import random
import cv2


# 归一化
def normalization(input_data, black_level, white_level):
    output_data = np.maximum(input_data.astype(float) - black_level, 0) / (white_level - black_level)
    return output_data


# 反归一化
def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


# 保存dng格式图像
def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


# 图像转化为四通道  返回 img (1736, 2312, 4) H 3472 W 4624
def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c, height, width


# 四通道图像还原为单通道 (3472, 4624)
def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[:, :, 2 * channel_y + channel_x]
    return output_data


# 生成噪声图像
def denoise_raw(input_path, output_path, black_level, white_level, noiseprofile_a, noiseprofile_b):
    """
    以下是我们如何生成模拟噪声配置文件以供参考
    here are how we generate simulated noise profile for your information
    """
    raw_data_expand_c, height, width = read_image(input_path)
    # 图像归一化 (1736, 2312, 4)
    raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)

    # (1736, 2312, 4) 根据a,b计算每一组的标准差
    raw_data_expand_c_normal_var = np.sqrt(noiseprofile_a * raw_data_expand_c_normal + noiseprofile_b)
    # 取方差
    # (1736, 2312, 4)  高斯分布 loc分布的均值, 分布的标准差(宽度)
    noise_data = np.random.normal(loc=raw_data_expand_c_normal,
                                  scale=raw_data_expand_c_normal_var,
                                  size=None)

    # 生成的噪声图反归一化
    noise_data = inv_normalization(noise_data, black_level, white_level)
    # 四通道图转化为单通道
    noise_data = write_image(noise_data, height, width)
    # 将生成的噪声图保存
    write_back_dng(input_path, output_path, noise_data)


def cal_noise_profile(test_dir, black_level, white_level):
    """
    your code should be given here
    """
    # 读取测试图像
    # img, h, w = read_image(test_dir)   # 不搞四通道
    noise_rd = np.array(rawpy.imread(test_dir).raw_image_visible).astype(np.float64)  # (3472, 4624)
    h = noise_rd.shape[0]
    w = noise_rd.shape[1]

    # gt_rd = cv2.GaussianBlur(noise_rd, (3, 3), 0)  # 高斯滤波
    gt_rd = signal.medfilt(noise_rd, (3, 3))  # 中值滤波
    # gt图归一化
    gt_rd_normal = normalization(gt_rd, black_level, white_level)
    # noise图像归一化
    noise_rd_normal = normalization(noise_rd, black_level, white_level)
    # 噪声图
    noise_diff = noise_rd_normal - gt_rd_normal
    # 切分图像
    pixel = 446
    point = []
    for x in range(h//pixel):
        for y in range(w//pixel):
            point.append([x, y])
    # 随机打乱顺序
    lt2 = [i for i in range(len(point))]

    a_out_list = []
    b_out_list = []
    # 迭代次数
    epochs = 2000
    for i in range(epochs):
        random.shuffle(lt2)
        # 随机抽取两张256×256的切片
        h_idx1 = point[lt2[0]][0]
        w_idx1 = point[lt2[0]][1]
        clip1_noise = noise_rd_normal[pixel * h_idx1:pixel * (h_idx1 + 1), pixel * w_idx1:pixel * (w_idx1 + 1)]
        clip1_diff = noise_diff[pixel * h_idx1:pixel * (h_idx1 + 1), pixel * w_idx1:pixel * (w_idx1 + 1)]

        h_idx2 = point[lt2[1]][0]
        w_idx2 = point[lt2[1]][1]
        clip2_noise = noise_rd_normal[pixel * h_idx2:pixel * (h_idx2 + 1), pixel * w_idx2:pixel * (w_idx2 + 1)]
        clip2_diff = noise_diff[pixel * h_idx2:pixel * (h_idx2 + 1), pixel * w_idx2:pixel * (w_idx2 + 1)]

        gt1 = float(np.mean(clip1_noise))  # 计算切片均值
        sigma1 = float(np.percentile(clip1_diff, 84.1344745)+np.percentile(clip1_diff, 97.724987))/3

        gt2 = float(np.mean(clip2_noise))
        sigma2 = float(np.percentile(clip2_diff, 84.1344745)+np.percentile(clip2_diff, 97.724987))/3

        a_out = (sigma1 ** 2 - sigma2 ** 2)/3.4/(gt1 - gt2)
        b_out = sigma1 ** 2 - a_out * gt1

        a_out_list.append(a_out)
        b_out_list.append(b_out)

    a_real = float(np.percentile(a_out_list, 50))
    a_last = np.clip(a_real, 0.0001, 0.01)
    b_real = float(np.percentile(b_out_list, 50))
    b_last = np.clip(b_real, 0.000001, 0.001)

    return a_last, b_last


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    test_dir = args.test_dir
    black_level = args.black_level
    white_level = args.white_level
    noiseprofile_a = args.noiseprofile_a
    noiseprofile_b = args.noiseprofile_b

    score = []

    for idx in range(len(noiseprofile_a)):
        """
        this part is an example showing how to generate simulated noise
        you do not need to modify this part
        """
        path = glob.glob(input_dir + '*.dng')
        # 生成噪声图
        for index in range(len(path)):
            in_path = path[index]
            in_basename = os.path.basename(in_path)
            input_path = input_dir + in_basename
            out_basename = in_basename.split(".")[0].strip()
            output_path = output_dir + out_basename + "_noise" + ".dng"
            denoise_raw(input_path, output_path, black_level, white_level, noiseprofile_a[idx], noiseprofile_b[idx])

        """
        this part aims to test your algorithm performance
        we will use multiple images and generate multiple noise profile para. to test your result
        three images in ./data/test/ is an example to help you understand our evaluation criteria
        you will not see ground truth test image and corresponding noise profile para. in test proc. 
        """
        # 测试生成图片分数
        path = glob.glob(test_dir + '*.dng')
        for index in range(len(path)):
            in_path = path[index]
            in_basename = os.path.basename(in_path)
            test_path = test_dir + in_basename

            """modify your function cal_noise_profile"""
            print("开始计算:  ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            a, b = cal_noise_profile(test_path, black_level, white_level)
            print("计算结束:  ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            evm_tmp = 0.5 * np.min([np.abs(a - noiseprofile_a[idx]) / noiseprofile_a[idx], 1]) \
                      + 0.5 * np.min([np.abs(b - noiseprofile_b[idx]) / noiseprofile_b[idx], 1])
            score_tmp = 100 - 100 * evm_tmp

            print("得分:", score_tmp)

            score.append(score_tmp)

    """your final score"""
    print('each score =', score)
    print('final score =', np.mean(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./data/gt/")
    parser.add_argument('--output_dir', type=str, default="./data/noise/")
    parser.add_argument('--test_dir', type=str, default="./data/noise/")
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)
    # a=[1e-4,1e-2] [0.0001, 0.01] [0.01,0.007,0.003,0.001,0.01,0.007,0.003,0.001,0.01,0.007,0.003,0.001,0.01,0.007,0.003,0.001]
    parser.add_argument('--noiseprofile_a', type=float, default=[random.uniform(0.0001,0.01)])
    # b=[1e-6,1e-3] [0.000001, 0.001] [0.001,0.0007,0.0003,0.0001,0.001,0.0007,0.0003,0.0001,0.001,0.0007,0.0003,0.0001,0.001,0.0007,0.0003,0.0001]
    parser.add_argument('--noiseprofile_b', type=float, default=[random.uniform(0.000001,0.001)])
    args = parser.parse_args()
    main(args)
