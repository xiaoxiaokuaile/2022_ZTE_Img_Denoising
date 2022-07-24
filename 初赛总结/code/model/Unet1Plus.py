# Learner: 王振强
# Learn Time: 2022/4/12 14:41
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class UNet1Plus(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, is_deconv=True, is_batchnorm=False):
        super(UNet1Plus, self).__init__()
        # 使用反卷积上采样
        self.is_deconv = is_deconv
        # 使用 BN 层
        self.is_batchnorm = is_batchnorm
        filters = [32, 64, 128, 256, 512]
        # filters = [64, 128, 256, 512, 1024]

        # downsampling
        self.conv1 = unetConv2(in_channels, filters[0], self.is_batchnorm)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

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
        conv1 = self.conv1(padded_image)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
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
        name = "./weights/UNet1Plus" + str(circle) + ".pth"
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
    test_input = torch.from_numpy(np.random.randn(1, 4, 434, 578)).float()
    net = UNet1Plus()
    output = net(test_input)
    print(output.shape)
    # print(net)

    # from thop import profile
    # input = torch.randn(1, 4, 512, 512)
    # flops, params = profile(net, inputs=(input,))  # 13396100.0
    # # 7760484 776W参数量
    # print(flops, params)



























