# Learner: 王振强
# Learn Time: 2022/4/27 21:26
import torch
import torch.nn as nn
import os
import torch.nn.functional as F


class InputCov(nn.Module):
    """
    处理原始图像
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.input_cov = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1, stride=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1, stride=(1, 1)),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=2, stride=(1, 1), dilation=(2,2)),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=2, stride=(1, 1), dilation=(2,2))
        )

    def forward(self, initial_data):
        input_cov = self.input_cov(initial_data)  # (batch,C,W,H)
        # 残差连接
        add_map = torch.sum(initial_data, dim=1)
        add_map_1 = torch.div(add_map, initial_data.shape[1])
        add_map_2 = add_map_1.unsqueeze(1)  # （batch,1,W,H）
        output_map = torch.add(input_cov, add_map_2)

        return output_map


class StdCovLocalResBlock(nn.Module):
    """
    使用标准卷积的局部残差模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.std_lrb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, stride=(1,1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1, stride=(1,1))
        )

    def forward(self, input_map):
        std_lrb = self.std_lrb(input_map)  # (btch,C,W,H)
        add_map = torch.sum(input_map, dim=1)  # (batch,W,H)
        add_map_1 = torch.div(add_map, input_map.shape[1]) #加和求均值
        add_map_2 = add_map_1.unsqueeze(1)  # （batch,1,W,H）
        output_map= torch.add(std_lrb , add_map_2) #特征融合
        return output_map


class DilCovLocalResBlock(nn.Module):
    """
    使用空洞卷积的局部残差模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dil_lrb = nn.Sequential(
            # dilation膨胀卷积系数, 覆盖(5,5)大小区域
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=2, stride=(1,1), dilation=(2,2)),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=2, stride=(1,1), dilation=(2,2))
        )

    def forward(self, input_map):
        dil_lrb= self.dil_lrb(input_map)  # (abtch,C,W,H)
        add_map = torch.sum(input_map, dim=1)  # (batch_size,W,H)
        add_map_1 = torch.div(add_map, input_map.shape[1])
        add_map_2 = add_map_1.unsqueeze(1)  # （batch,1,W,H）
        output_map= torch.add(dil_lrb, add_map_2)
        return output_map


class LeftGlobalResBlock(nn.Module):
    """
    网络左半部分的全局残差模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.std_cov = StdCovLocalResBlock(in_channels, out_channels)
        self.dil_cov = DilCovLocalResBlock(out_channels, out_channels)

    def forward(self, down_map):
        # 残差正常卷积
        std_cov = self.std_cov(down_map)
        # 残差空洞卷积
        dil_cov = self.dil_cov(std_cov)
        add_map = torch.sum(down_map, dim=1)
        add_map_1 = torch.div(add_map, down_map.shape[1])
        add_map_2 = add_map_1.unsqueeze(1)  # （batch,1,W,H）
        output_map= torch.add(dil_cov, add_map_2)
        return output_map


class RightGlobalResBlock(nn.Module):
    """
    网络右半部分的全局残差模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.std_cov_1 = StdCovLocalResBlock(in_channels, out_channels)
        self.std_cov_2 = StdCovLocalResBlock(out_channels, out_channels)

    def forward(self, up_map):
        std_cov_1 = self.std_cov_1(up_map)
        std_cov_2 = self.std_cov_2(std_cov_1)
        add_map = torch.sum(up_map, dim=1)
        add_map_1 = torch.div(add_map, up_map.shape[1])
        add_map_2 = add_map_1.unsqueeze(1)  # （batch,1,W,H）
        output_map= torch.add(std_cov_2, add_map_2)
        return output_map


class Down(nn.Module):
    """
    下采样模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=(2,2), padding=0, stride=(2,2))

    def forward(self, input_map):
        return self.down(input_map)


class Up(nn.Module):
    """
    上采样模块
    """

    def __init__(self):
        super().__init__()
        self.up = nn.PixelShuffle(2)

    def forward(self, input_map, skip_map):
        up = self.up(input_map)
        up_map = torch.sum(up, dim=1)
        up_map_1 = torch.div(up_map, input_map.shape[1])
        up_map_2 = up_map_1.unsqueeze(1)  # （batch,1,W,H）
        output_map= torch.add(skip_map, up_map_2)
        return output_map


class OutputCov(nn.Module):
    """
    输出
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cov = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, stride=(1,1))

    def forward(self, input_map):
        return self.cov(input_map)


class DRUNet(nn.Module):
    def __init__(self):
        super(DRUNet, self).__init__()
        filters = [48, 96, 192, 384]  # [64, 128, 256, 512]
        self.input = InputCov(4, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.left1 = LeftGlobalResBlock(filters[1], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.left2 = LeftGlobalResBlock(filters[2], filters[2])
        self.down3 = Down(filters[2], filters[3])
        self.left3 = LeftGlobalResBlock(filters[3], filters[3])
        self.up1 = Up()
        self.right1 = RightGlobalResBlock(filters[2], filters[2])
        self.up2 = Up()
        self.right2 = RightGlobalResBlock(filters[1], filters[1])
        self.up3 = Up()
        self.right3 = RightGlobalResBlock(filters[0], filters[0])
        self.output = OutputCov(filters[0], 4)

    def forward(self, x):
        n, c, h, w = x.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), 'replicate')

        input_map = self.input(x)
        down1_map = self.down1(input_map)
        left1_map = self.left1(down1_map)
        down2_map = self.down2(left1_map)
        left2_map = self.left2(down2_map)
        down3_map = self.down3(left2_map)
        left3_map = self.left3(down3_map)
        up1_map = self.up1(left3_map, left2_map)
        right1_map = self.right1(up1_map)
        up2_map = self.up2(right1_map, left1_map)
        right2_map = self.right2(up2_map)
        up3_map = self.up3(right2_map, input_map)
        right3_map = self.right3(up3_map)
        output = self.output(right3_map)

        output = output[:, :, :h, :w]

        return output

    def save(self, circle):
        name = "./weights/DRUnet" + str(circle) + ".pth"
        torch.save(self.state_dict(), name)

    # 加载模型
    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(weight_path))
            else:
                self.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print("load %s success!" % weight_path)
        else:
            print("%s do not exists." % weight_path)


if __name__ == '__main__':
    net = DRUNet()
    input_ = torch.Tensor(1, 4, 512, 512)
    out = net(input_)
    print(out.shape)

    from thop import profile
    input = torch.randn(1, 4, 256, 256)
    flops, params = profile(net, inputs=(input,))  # 13396100.0
    #  16326209.0
    print(flops, params)
