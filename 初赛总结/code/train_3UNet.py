import os
import time
import pandas as pd
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
from model import UUnet_B0,UUnet_B3

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
    criterion = MSELoss().to(device)
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
        loss_1 = AverageMeter()
        loss_2 = AverageMeter()
        for circle, (x, label, _) in enumerate(trainDataLoader, 0):
            # 加载数据
            x = x.to(device)           # 归一化后的x [bz, 4, 434, 578]
            label = label.to(device)   # 归一化后的label [bz, 4, 434, 578]

            output1, output2, output3, output4, out_left, out_right = model(x)
            # 左分支
            left_net = ((output1 + output2 + output3) / 3 + out_left) / 2
            # 右分支
            right_net = ((output2 + output3 + output4) / 3 + out_right) / 2
            # 分支合并
            out_add = (left_net + right_net) / 2

            loss_left = criterion(left_net, label)
            loss_right = criterion(right_net, label)
            loss_all = criterion(out_add, label)
            loss = loss_all

            optimizer.zero_grad()  # 后向传播
            loss.backward()

            loss_1.add(loss_left.item())
            loss_2.add(loss_right.item())
            loss_meter.add(loss.item())

            optimizer.step()       # 更新模型
        if True:
            scheduler.step()
            # ------------------ 验证集 -------------------
            model.eval()
            test_loss = AverageMeter()
            test_loss_1 = AverageMeter()
            test_loss_2 = AverageMeter()
            for circle, (x, label, _) in enumerate(valDataLoader, 0):
                x = x.to(device)
                label = label.to(device)

                output1, output2, output3, output4, out_left, out_right = model(x)
                # 左分支
                left_net = ((output1 + output2 + output3) / 3 + out_left)/2
                # 右分支
                right_net = ((output2 + output3 + output4) / 3 + out_right) / 2
                # 分支合并
                out_add = (left_net + right_net)/2

                loss_left = criterion(left_net, label)
                loss_right = criterion(right_net, label)
                loss_all = criterion(out_add, label)
                loss = loss_all

                test_loss_1.add(loss_left.item())
                test_loss_2.add(loss_right.item())
                test_loss.add(loss.item())
            # ---------------------------------------------
            # 保存最优模型
            if best_loss > test_loss.avg:
                best_loss = test_loss.avg
                # 保存最好的模型
                model.save("_best")
            model.save('_last')
            # -------------- 每个epoch输出结果 ---------------
            # 输出训练日志
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),end='')
            print("epoch:[%02d/%02d] | Lr: %.6f | Train loss %.5f | Test loss %.5f | Best loss %.5f " % \
                  (epoch,totalEpoch, scheduler.get_last_lr()[0],loss_meter.avg*100,test_loss.avg*100, best_loss*100))
            print("                  | Train loss_left: %.5f | Train loss_right:%.5f | Test  loss_left: %.5f | Test loss_right:%.5f " % \
                  (loss_1.avg*100,loss_2.avg*100,test_loss_1.avg*100,test_loss_2.avg*100))


if __name__ == '__main__':
    net_list = {
                # 'UUnet_B0': UUnet_B0(),
                'UUnet_B3': UUnet_B3(),
                }
    net = net_list['UUnet_B3']
    # 加载预训练模型
    # net.load_model("./weights/unet_best.pth")
    train(net)
