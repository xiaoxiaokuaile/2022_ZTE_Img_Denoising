import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from utils import AverageMeter
# 加载超参数
from config.config import *
# 加载数据用函数
from utils.dataset import Captcha
from torch.utils.data import DataLoader
from utils.utils import normalization,inv_normalization,read_image,write_image
# loss
from torch.nn import MSELoss
from model import Unet_baseline,Unet_mini

"""
    预测代码
"""

# 加载数据
def dataloader():
    train_fold = pd.read_csv(train_fold_csv)
    train_csv = train_fold[train_fold['fold'] != 0][['train', 'label']]
    val_csv = train_fold[train_fold['fold'] == 0][['train', 'label']]
    # 训练集训练
    trainDataset = Captcha(train_csv, input_path, label_path, data_mode='train')
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=4)
    # 验证集
    valDataset = Captcha(val_csv, input_path, label_path, data_mode='val')
    valDataLoader = DataLoader(valDataset, batch_size=1, shuffle=False, num_workers=4)

    return trainDataLoader,valDataLoader


def Score(model):
    # 设备参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # loss 损失函数
    criterion = MSELoss().to(device)
    # 加载数据
    trainDataLoader,valDataLoader = dataloader()

    # ------------------ 验证集准确率 -------------------
    # 记录loss信息
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    for circle, (x, label, label_int) in enumerate(valDataLoader, 0):
        x = x.to(device)
        label = label.to(device)
        label_int = label_int.to(device)
        output = model(x)
        # 计算损失
        loss = criterion(output, label)
        test_loss.add(loss.item())
        # # 计算准确率
        t_acc = calculat_acc(output, label_int)
        test_acc.add(float(t_acc))
        print(float(loss),t_acc)
    print(test_loss.avg,test_acc.avg)


# 计算得分
def calculat_acc(output,label):
    # post-process  后处理
    result_data = output.cpu().detach().numpy().transpose(0, 2, 3, 1) # (1, 1736, 2312, 4)
    label = label.cpu().detach().numpy().squeeze(axis=0)  # (1, 3472, 4624)
    # 将网络输出结果转化为 int16 类型
    result_data = inv_normalization(result_data, black_level, white_level) # (1, 1736, 2312, 4)
    # 图像拼接
    img_pj = write_image(result_data, height, width)  # (3472, 4624)

    # PSNR峰值信噪比 44.39486745435859
    psnr = PSNR(label, img_pj.astype(np.float32), data_range=white_level)
    # 结构相似性 0.9893065017823934
    ssim = SSIM(label, img_pj.astype(np.float32), multichannel=True, data_range=white_level)
    w = 0.8
    psnr_max = 60
    psnr_min = 30
    ssim_min = 0.8
    score = (w * max(psnr - psnr_min, 0) / (psnr_max - psnr_min) + (1 - w) * max(ssim - ssim_min, 0) / (1 - ssim_min)) * 100

    return score

if __name__ == '__main__':
    # -------------- 单模型预测 -------------
    model_pr = Unet_baseline()
    model_pr.load_model("./weights/unet_best.pth")
    model_pr.eval()
    Score(model_pr)



