# Learner: 王振强
# Learn Time: 2022/5/7 20:12
from utils.utils import write_back_dng
import rawpy
import os
import numpy as np


"""
    模型融合
"""

def model_ad(model1_path, model2_path, save_path):
    for i in range(10):
        dng_path1 = os.path.join(model1_path,"denoise" + str(i) + ".dng")
        dng_path2 = os.path.join(model2_path,"denoise" + str(i) + ".dng")
        read_dng1 = rawpy.imread(dng_path1).raw_image_visible  # (3472, 4624)
        read_dng2 = rawpy.imread(dng_path2).raw_image_visible  # (3472, 4624)
        # print(read_dng1)
        # 模型融合
        dng_add = np.array((read_dng1 + read_dng2)/2).astype(np.int16)
        # print(dng_add)
        # 将拼接的图像转化为dng格式
        dest_path = os.path.join(save_path, "denoise" + str(i) + ".dng")
        write_back_dng(dng_path1, dest_path, dng_add)

if __name__ == '__main__':
    model1_path = r'F:\JS\2022中兴捧月\提交过程\58.45分'
    model2_path = r'F:\JS\2022中兴捧月\提交过程\MPRNet数据增强L1'
    save_path = r'F:\JS\2022中兴捧月\提交过程\58.4_58.84'

    model_ad(model1_path, model2_path, save_path)





















