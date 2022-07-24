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
from model import Unet,UNet1Plus,UNet2Plus,UNet3Plus
from model import B0_UNet1,B3_UNet1
from model import B0_UNet2,B3_UNet2
from model import ResNet18_Unet2,ResNet34_Unet2,ResNet50_Unet2

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
    valDataLoader = DataLoader(valDataset, batch_size=4, shuffle=False, num_workers=4)

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
        loss_3 = AverageMeter()
        loss_4 = AverageMeter()
        for circle, (x, label, _) in enumerate(trainDataLoader, 0):
            # 加载数据
            x = x.to(device)           # 归一化后的x [bz, 4, 434, 578]
            label = label.to(device)   # 归一化后的label [bz, 4, 434, 578]

            output1,output2,output3,output4 = model(x)
            output = (output1 + output2 + output3 + output4) / 4
            loss1 = criterion(output1, label)
            loss2 = criterion(output2, label)
            loss3 = criterion(output3, label)
            loss4 = criterion(output4, label)
            loss_avg = criterion(output, label)
            loss = loss_avg

            optimizer.zero_grad()  # 后向传播
            loss.backward()

            loss_1.add(loss1.item())
            loss_2.add(loss2.item())
            loss_3.add(loss3.item())
            loss_4.add(loss4.item())
            loss_meter.add(loss.item())

            optimizer.step()       # 更新模型
        if True:
            scheduler.step()
            # ------------------ 验证集 -------------------
            model.eval()
            test_loss = AverageMeter()
            test_loss_1 = AverageMeter()
            test_loss_2 = AverageMeter()
            test_loss_3 = AverageMeter()
            test_loss_4 = AverageMeter()
            for circle, (x, label, _) in enumerate(valDataLoader, 0):
                x = x.to(device)
                label = label.to(device)

                output1, output2, output3, output4 = model(x)
                output = (output1 + output2 + output3 + output4)/4
                loss1 = criterion(output1, label)
                loss2 = criterion(output2, label)
                loss3 = criterion(output3, label)
                loss4 = criterion(output4, label)
                loss_avg = criterion(output, label)
                loss = loss_avg

                test_loss_1.add(loss1.item())
                test_loss_2.add(loss2.item())
                test_loss_3.add(loss3.item())
                test_loss_4.add(loss4.item())
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
            print("                  | Train loss1: %.5f | Train loss2:%.5f | Train loss3:%.5f | Train loss4:%.5f" % \
                  (loss_1.avg*100,loss_2.avg*100,loss_3.avg*100,loss_4.avg*100))
            print("                  | Test  loss1: %.5f | Test  loss2:%.5f | Test  loss3:%.5f | Test  loss4:%.5f" % \
                  (test_loss_1.avg*100,test_loss_2.avg*100,test_loss_3.avg*100,test_loss_4.avg*100))


if __name__ == '__main__':
    net_list = {
                # 'Unet': Unet(),
                # 'UNet1Plus': UNet1Plus(),
                # 'UNet1Plus_BN': UNet1Plus_BN(),
                # 'UNet2Plus': UNet2Plus(),
                # 'UNet2Plus_BN': UNet2Plus_BN(),
                # 'UNet3Plus': UNet3Plus(),
                # 'B0_UNet1': B0_UNet1(),
                # 'B1_UNet1': B1_UNet1(),
                # 'B2_UNet1': B2_UNet1(),
                # 'B3_UNet1': B3_UNet1(),
                # 'B0_UNet2': B0_UNet2(),
                # 'B1_UNet2': B1_UNet2(),
                # 'B2_UNet2': B2_UNet2(),
                # 'B3_UNet2': B3_UNet2(),
                # 'B4_UNet2': B4_UNet2(),
                # 'B5_UNet2': B5_UNet2(),
                'ResNet18_Unet2': ResNet18_Unet2(),
                # 'ResNet34_Unet2': ResNet34_Unet2(),
                # 'ResNet50_Unet2': ResNet50_Unet2(),
                # 'B0_UNet3': B0_UNet3(),
                # 'B1_UNet3': B1_UNet3(),
                # 'B2_UNet3': B2_UNet3(),
                # 'B3_UNet3': B3_UNet3(),
                }
    net = net_list['ResNet18_Unet2']
    # 加载预训练模型
    net.load_model("./weights/ResNet18_UNet2_best.pth")
    train(net)
