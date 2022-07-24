import os
import time
import pandas as pd
import numpy as np
import torch
from utils import AverageMeter
# 加载超参数
from config.config import *
# 加载数据用函数
from utils.dataset import Captcha
from torch.utils.data import DataLoader
# 优化器
from torch.optim import Adam
# loss
from torch.nn import MSELoss
from utils.losses import CharbonnierLoss,EdgeLoss
# 学习率衰减策略
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
# model
from model import Restormer,DRUNet,MPRNet

torch.manual_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 加载数据
def dataloader():
    train_fold = pd.read_csv(train_fold_csv)
    train_csv = train_fold[train_fold['fold'] != -1][['train', 'label']]
    val_csv = train_fold[train_fold['fold'] == 0][['train', 'label']]
    # 训练集训练
    trainDataset = Captcha(train_csv, input_path, label_path, data_mode='train')
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=4)
    # 验证集
    valDataset = Captcha(val_csv, input_path, label_path, data_mode='val')
    valDataLoader = DataLoader(valDataset, batch_size=8, shuffle=False, num_workers=4)

    return trainDataLoader,valDataLoader


def train(model):
    # 设备参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # loss 损失函数
    L1 = CharbonnierLoss().to(device)  # 类L1损失函数
    L2 = MSELoss().to(device)
    # 获取优化器, 学习率衰减策略
    optimizer = Adam(model.parameters(), lr=learningRate)
    scheduler = CosineAnnealingLR(optimizer,totalEpoch, eta_min=1e-6, last_epoch=-1)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.3)
    # 加载数据
    trainDataLoader,valDataLoader = dataloader()

    # 训练Model
    best_loss = 1
    for epoch in range(totalEpoch):
        model.train()
        loss_meter = AverageMeter()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for circle, (x, label, _) in enumerate(trainDataLoader, 0):
            # 加载数据
            x = x.to(device)           # 归一化后的x [bz, 4, 434, 578]
            label = label.to(device)   # 归一化后的label [bz, 4, 434, 578]

            output3, output2, output1 = model(x)
            # torch.clamp(output[i], 0, 1) 将数据截取到0-1之间
            loss3 = L1(output3, label)
            loss2 = L1(output2, label)
            loss1 = L1(output1, label)
            loss = (loss3 + loss2 + loss1)/3

            optimizer.zero_grad()  # 后向传播
            loss.backward()
            loss3_meter.add(loss3.item())
            loss2_meter.add(loss2.item())
            loss1_meter.add(loss1.item())
            loss_meter.add(loss.item())
            optimizer.step()       # 更新模型

            # -------------- 输出结果 ---------------
            if circle % 500 == 0:
                # 输出训练日志
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),end='')
                print("epoch:[%02d/%02d] | step:[%02d]| Lr: %.6f | Train loss %.5f | Best loss %.5f " % \
                      (epoch,totalEpoch, circle, scheduler.get_last_lr()[0],loss_meter.avg*100, best_loss*100))
                print("                  | Train loss3 %.5f | Train loss2 %.5f | Train loss1 %.5f " % \
                      (loss3_meter.avg*100,loss2_meter.avg*100,loss1_meter.avg*100))
        # 保存最优模型
        if best_loss > loss_meter.avg:
            best_loss = loss_meter.avg
            # 保存最好的模型
            model.save("_best")
        model.save('_last')

        if True:
            scheduler.step()
            # ------------------ 验证集 -------------------
            # model.eval()
            # test_loss = AverageMeter()
            # for circle, (x, label, _) in enumerate(valDataLoader, 0):
            #     x = x.to(device)
            #     label = label.to(device)
            #
            #     output = model(x)
            #     loss = criterion(output, label)
            #
            #     test_loss.add(loss.item())
            # ---------------------------------------------


if __name__ == '__main__':
    net_list = {
                # 'Restormer': Restormer(),
                # 'DRUNet': DRUNet(),
                'MPRNet': MPRNet(),
                }
    net = net_list['MPRNet']
    # 加载预训练模型
    net.load_model("./weights/MPRNet_best.pth")
    train(net)


