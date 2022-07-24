# Learner: 王振强
# Learn Time: 2022/4/12 14:41
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import os

"""
    Unet官方结构:
    4->64->64   --------------------------------------------------------------------->   [64|64]->64->64->4
            ↓                                                                                 ↑
           64->128->128   ----------------------------------------------->   [128|128]->128->128
                     ↓                                                             ↑
                    128->256->256   -------------------------->   [256|256]->256->256
                               ↓                                        ↑
                              256->512->512   ----->   [512|512]->512->512
                                         ↓                   ↑ (反卷积通道减半)
                                        512->1024 --- 1024->1024
"""


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=False, ks=(3,3), stride=(1,1), padding=1):
        super(unetConv2, self).__init__()
        # 是否添加BN层
        if is_batchnorm:
            self.conv = nn.Sequential(
                                 nn.Conv2d(in_size, out_size, ks, stride, padding),
                                 nn.BatchNorm2d(out_size),
                                 nn.LeakyReLU(negative_slope=0.125),
                                 nn.Conv2d(out_size, out_size, ks, stride, padding),
                                 nn.BatchNorm2d(out_size),
                                 nn.LeakyReLU(negative_slope=0.125),
                                 )
        else:
            self.conv = nn.Sequential(
                                 nn.Conv2d(in_size, out_size, ks, stride, padding),
                                 nn.LeakyReLU(negative_slope=0.125),
                                 nn.Conv2d(out_size, out_size, ks, stride, padding),
                                 nn.LeakyReLU(negative_slope=0.125),
                                 )

    def forward(self, inputs):
        out = self.conv(inputs)
        return out


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(out_size * 2, out_size, False)
        # 是否反卷积
        if is_deconv:
            # kernel_size=4, stride=2, padding=1   或者 kernel_size=2, stride=2, padding=0
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=(2,2), stride=(2,2))
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class B0_UNet1(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, is_deconv=True, is_batchnorm=False):
        super(B0_UNet1, self).__init__()
        model = EfficientNet.from_pretrained('efficientnet-b0',weights_path='./model/PreModel/efficientnet-b0-355c32eb.pth')

        self.cnn2 = nn.Sequential(
            model._blocks[1],  # 24 (2,2) torch.Size([1, 24, 128, 128])
            model._blocks[2],  # 24
        )
        self.cnn3 = nn.Sequential(
            model._blocks[3],  # 40 (2,2) torch.Size([1, 40, 64, 64])
            model._blocks[4],  # 40
        )
        self.cnn4 = nn.Sequential(
            model._blocks[5],  # 80 (2,2) torch.Size([1, 80, 32, 32])
            model._blocks[6],  # 80
            model._blocks[7],  # 80
            model._blocks[8],  # 112
            model._blocks[9],  # 112
            model._blocks[10],  # 112
        )
        self.cnn5 = nn.Sequential(
            model._blocks[11],  # 192 (2,2) torch.Size([1, 192, 16, 16])
            model._blocks[12],  # 192
            model._blocks[13],  # 192
            model._blocks[14],  # 192
            model._blocks[15],  # 320
        )
        # 使用反卷积上采样
        self.is_deconv = is_deconv
        # 使用 BN 层
        self.is_batchnorm = is_batchnorm
        filters = [16, 24, 40, 112, 320]

        # downsampling
        self.conv1 = unetConv2(in_channels, filters[0], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        self.outconv1 = nn.Conv2d(filters[0], out_channels, kernel_size=(1,1), stride=(1,1))

    def forward(self, inputs):
        n, c, h, w = inputs.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_image = F.pad(inputs, (0, w_pad, 0, h_pad), 'replicate')
        # down
        conv1 = self.conv1(padded_image)  # torch.Size([1, 16, 512, 512])
        conv2 = self.cnn2(conv1)          # torch.Size([1, 24, 256, 256])
        conv3 = self.cnn3(conv2)          # torch.Size([1, 48, 128, 128])
        conv4 = self.cnn4(conv3)          # torch.Size([1, 120, 64, 64])
        center = self.cnn5(conv4)         # torch.Size([1, 352, 32, 32])
        # up
        up4 = self.up_concat4(center, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)
        # out
        out = self.outconv1(up1)
        out = out[:, :, :h, :w]

        return out

    def save(self, circle):
        name = "./weights/B0_UNet1" + str(circle) + ".pth"
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


if __name__ == "__main__":
    import numpy as np
    test_input = torch.from_numpy(np.random.randn(1, 4, 512, 512)).float()
    net = B0_UNet1()
    output = net(test_input)
    print(output.shape)
    # print(net)

    # from thop import profile
    # input = torch.randn(1, 4, 512, 512)
    # flops, params = profile(net, inputs=(input,))  # 13396100.0
    # # 7760484 776W参数量
    # print(flops, params)

    # test_input = torch.from_numpy(np.random.randn(1, 3, 512, 512)).float()
    # model = EfficientNet.from_pretrained('efficientnet-b0',weights_path='./PreModel/efficientnet-b0-355c32eb.pth')
    # model1 = nn.Sequential(
    #     model._conv_stem,   # 32 (2,2) torch.Size([1, 32, 256, 256])
    #     model._bn0,         # 32
    #     model._blocks[0],   # 16
    #     model._blocks[1],   # 24 (2,2) torch.Size([1, 24, 128, 128])
    #     model._blocks[2],   # 24
    #     model._blocks[3],   # 40 (2,2) torch.Size([1, 40, 64, 64])
    #     model._blocks[4],   # 40
    #     model._blocks[5],   # 80 (2,2) torch.Size([1, 80, 32, 32])
    #     model._blocks[6],   # 80
    #     model._blocks[7],   # 80
    #     model._blocks[8],   # 112
    #     model._blocks[9],   # 112
    #     model._blocks[10],  # 112
    #     model._blocks[11],  # 192 (2,2) torch.Size([1, 192, 16, 16])
    #     model._blocks[12],  # 192
    #     model._blocks[13],  # 192
    #     model._blocks[14],  # 192
    #     model._blocks[15],  # 320
    #     # model._conv_head,   # (1280,7,7)
    #     # model._bn1,
    #     # model._avg_pooling, # (1280,1,1)
    #     # model._dropout,     # Dropout(p=0.2, inplace=False)
    #     # model._fc,
    #     # model._swish
    # )
    # out2 = model1(test_input)
    # print(out2.shape)




























