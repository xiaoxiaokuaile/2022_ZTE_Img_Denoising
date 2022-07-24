# Learner: 王振强
# Learn Time: 2022/4/12 12:52
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from efficientnet_pytorch import EfficientNet
import os

"""
    给Unet++两边加两个Unet分支辅助
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


class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        self.conv = unetConv2(n_concat * out_size, out_size, False)
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


# 两个相同大小的tensor合并为一个tensor
class concat_chanel(nn.Module):
    def __init__(self, out_size, n_concat=2):
        super(concat_chanel, self).__init__()
        # 调整通道数
        self.conv = nn.Conv2d(n_concat * out_size, out_size, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, tensor1, tensor2):
        outputs0 = torch.cat([tensor1, tensor2], 1)
        return self.conv(outputs0)


class UUnet_B0(nn.Module):
    def __init__(self, in_channels=4, out_channels=4,is_deconv=True, is_batchnorm=False, out_num=False):
        super(UUnet_B0, self).__init__()
        # 使用反卷积 还是 上采样
        self.is_deconv = is_deconv
        # 是否使用BN层
        self.is_batchnorm = is_batchnorm
        # 返回4个输出或最后一个输出
        self.out_num = out_num
        filters = [16, 24, 40, 112, 320]
        # downsampling
        model = EfficientNet.from_pretrained('efficientnet-b0', weights_path='./model/PreModel/efficientnet-b0-355c32eb.pth')
        self.conv00 = unetConv2(in_channels, filters[0], self.is_batchnorm)
        self.conv10 = nn.Sequential(
            model._blocks[1],  # 24 (2,2) torch.Size([1, 24, 128, 128])
            model._blocks[2],  # 24
        )
        self.conv20 = nn.Sequential(
            model._blocks[3],  # 40 (2,2) torch.Size([1, 40, 64, 64])
            model._blocks[4],  # 40
        )
        self.conv30 = nn.Sequential(
            model._blocks[5],  # 80 (2,2) torch.Size([1, 80, 32, 32])
            model._blocks[6],  # 80
            model._blocks[7],  # 80
            model._blocks[8],  # 112
            model._blocks[9],  # 112
            model._blocks[10],  # 112
        )
        self.conv40 = nn.Sequential(
            model._blocks[11],  # 192 (2,2) torch.Size([1, 192, 16, 16])
            model._blocks[12],  # 192
            model._blocks[13],  # 192
            model._blocks[14],  # 192
            model._blocks[15],  # 320
        )

        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv, 2)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv, 2)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv, 2)
        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv, 2)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], out_channels, kernel_size=(1,1), stride=(1,1))
        self.final_2 = nn.Conv2d(filters[0], out_channels, kernel_size=(1,1), stride=(1,1))
        self.final_3 = nn.Conv2d(filters[0], out_channels, kernel_size=(1,1), stride=(1,1))
        self.final_4 = nn.Conv2d(filters[0], out_channels, kernel_size=(1,1), stride=(1,1))

        # 左分支
        self.conv_left_1 = unetConv2(in_channels, filters[0], self.is_batchnorm)  # (1, 16, 512, 512)
        self.cat_left_1 = concat_chanel(filters[0])  # concat特征, 合并 分支特征图 及 主干特征图, 并调整通道数
        self.left_maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv_left_2 = unetConv2(filters[0], filters[1], self.is_batchnorm)  # (1, 24, 256, 256)
        self.cat_left_2 = concat_chanel(filters[1])
        self.left_maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv_left_3 = unetConv2(filters[1], filters[2], self.is_batchnorm)  # (1, 40, 128, 128)
        self.cat_left_3 = concat_chanel(filters[2])
        self.left_maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv_left_4 = unetConv2(filters[2], filters[3], self.is_batchnorm)  # (1, 112, 64, 64)
        self.cat_left_4 = concat_chanel(filters[3])
        self.left_maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.up_left_3 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_left_2 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_left_1 = unetUp_origin(filters[1], filters[0], self.is_deconv)

        self.out_left_conv1 = nn.Conv2d(filters[0], out_channels, kernel_size=(1, 1), stride=(1, 1))

        # 右分支
        self.conv_right_1 = unetConv2(in_channels, filters[0], self.is_batchnorm)  # (1, 16, 512, 512)
        self.cat_right_1 = concat_chanel(filters[0])  # concat特征, 合并 分支特征图 及 主干特征图, 并调整通道数
        self.right_maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv_right_2 = unetConv2(filters[0], filters[1], self.is_batchnorm)  # (1, 24, 256, 256)
        self.cat_right_2 = concat_chanel(filters[1])
        self.right_maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv_right_3 = unetConv2(filters[1], filters[2], self.is_batchnorm)  # (1, 40, 128, 128)
        self.cat_right_3 = concat_chanel(filters[2])
        self.right_maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv_right_4 = unetConv2(filters[2], filters[3], self.is_batchnorm)  # (1, 112, 64, 64)
        self.cat_right_4 = concat_chanel(filters[3])
        self.right_maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.up_right_3 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_right_2 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_right_1 = unetUp_origin(filters[1], filters[0], self.is_deconv)

        self.out_right_conv1 = nn.Conv2d(filters[0], out_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inputs):
        n, c, h, w = inputs.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_image = F.pad(inputs, (0, w_pad, 0, h_pad), 'replicate')
        # column : 0
        # 下采样
        X_00 = self.conv00(padded_image)  # torch.Size([1, 16, 512, 512])
        X_10 = self.conv10(X_00)          # torch.Size([1, 24, 256, 256])
        X_20 = self.conv20(X_10)          # torch.Size([1, 40, 128, 128])
        X_30 = self.conv30(X_20)          # torch.Size([1, 112, 64, 64])
        X_40 = self.conv40(X_30)          # torch.Size([1, 320, 32, 32])
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)  # torch.Size([1, 16, 512, 512])
        X_11 = self.up_concat11(X_20, X_10)  # torch.Size([1, 24, 256, 256])
        X_21 = self.up_concat21(X_30, X_20)  # torch.Size([1, 40, 128, 128])
        X_31 = self.up_concat31(X_40, X_30)  # torch.Size([1, 112, 64, 64])
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)  # torch.Size([1, 16, 512, 512])
        X_12 = self.up_concat12(X_21, X_10, X_11)  # torch.Size([1, 24, 256, 256])
        X_22 = self.up_concat22(X_31, X_20, X_21)  # torch.Size([1, 40, 128, 128])
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)  # torch.Size([1, 16, 512, 512])
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)  # torch.Size([1, 24, 256, 256])
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)  # torch.Size([1, 16, 512, 512])

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)
        final_1 = final_1[:, :, :h, :w]
        final_2 = final_2[:, :, :h, :w]
        final_3 = final_3[:, :, :h, :w]
        final_4 = final_4[:, :, :h, :w]
        final = (final_1 + final_2 + final_3 + final_4) / 4

        # 浅层分支
        input1 = (final_1 + final_2 + final_3)/3  # torch.Size([1, 4, 512, 512])
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_left = F.pad(input1, (0, w_pad, 0, h_pad), 'replicate')
        # down
        conv1_left = self.conv_left_1(padded_left)    # torch.Size([1, 16, 512, 512])
        cat1_left = self.cat_left_1(conv1_left,X_00)  # torch.Size([1, 16, 512, 512])
        maxpool1_left = self.left_maxpool1(cat1_left)      # torch.Size([1, 16, 256, 256])

        conv2_left = self.conv_left_2(maxpool1_left)       # torch.Size([1, 24, 256, 256])
        cat2_left = self.cat_left_2(conv2_left,X_10)  # torch.Size([1, 24, 256, 256])
        maxpool2_left = self.left_maxpool2(cat2_left)      # torch.Size([1, 40, 128, 128])

        conv3_left = self.conv_left_3(maxpool2_left)       # torch.Size([1, 40, 128, 128])
        cat3_left = self.cat_left_3(conv3_left,X_20)  # torch.Size([1, 40, 128, 128])
        maxpool3_left = self.left_maxpool3(cat3_left)      # torch.Size([1, 40, 64, 64])

        conv4_left = self.conv_left_4(maxpool3_left)      # torch.Size([1, 112, 64, 64])
        cat4_left = self.cat_left_4(conv4_left,X_30) # torch.Size([1, 112, 64, 64])
        # up
        up3_left = self.up_left_3(cat4_left, cat3_left)  # torch.Size([1, 40, 128, 128])
        up2_left = self.up_left_2(up3_left, cat2_left)   # torch.Size([1, 24, 256, 256])
        up1_left = self.up_left_1(up2_left, cat1_left)   # torch.Size([1, 16, 512, 512])
        # out
        out_left = self.out_left_conv1(up1_left)         # torch.Size([1, 4, 512, 512])
        out_left = out_left[:, :, :h, :w]                # torch.Size([1, 4, 512, 512])
        # 深层分支
        input2 = (final_2 + final_3 + final_4)/3
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_right = F.pad(input2, (0, w_pad, 0, h_pad), 'replicate')
        # down
        conv1_right = self.conv_left_1(padded_right)    # torch.Size([1, 16, 512, 512])
        cat1_right = self.cat_left_1(conv1_right,X_04)  # torch.Size([1, 16, 512, 512])
        maxpool1_right = self.left_maxpool1(cat1_right)      # torch.Size([1, 16, 256, 256])

        conv2_right = self.conv_left_2(maxpool1_right)       # torch.Size([1, 24, 256, 256])
        cat2_right = self.cat_left_2(conv2_right,X_13)  # torch.Size([1, 24, 256, 256])
        maxpool2_right = self.left_maxpool2(cat2_right)      # torch.Size([1, 40, 128, 128])

        conv3_right = self.conv_left_3(maxpool2_right)       # torch.Size([1, 40, 128, 128])
        cat3_right = self.cat_left_3(conv3_right,X_22)  # torch.Size([1, 40, 128, 128])
        maxpool3_right = self.left_maxpool3(cat3_right)      # torch.Size([1, 40, 64, 64])

        conv4_right = self.conv_left_4(maxpool3_right)      # torch.Size([1, 112, 64, 64])
        cat4_right = self.cat_left_4(conv4_right,X_31) # torch.Size([1, 112, 64, 64])
        # up
        up3_right = self.up_left_3(cat4_right, cat3_right)  # torch.Size([1, 40, 128, 128])
        up2_right = self.up_left_2(up3_right, cat2_right)   # torch.Size([1, 24, 256, 256])
        up1_right = self.up_left_1(up2_right, cat1_right)   # torch.Size([1, 16, 512, 512])
        # out
        out_right = self.out_left_conv1(up1_right)         # torch.Size([1, 4, 512, 512])
        out_right = out_right[:, :, :h, :w]                # torch.Size([1, 4, 512, 512])

        if self.out_num:
            return final
        else:
            return final_1,final_2,final_3,final_4,out_left,out_right

    def save(self, circle):
        name = "./weights/UUnet_B0" + str(circle) + ".pth"
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
    net = UUnet_B0()
    # input = torch.from_numpy(np.random.randn(1, 4, 512, 512)).float()
    # output = net(input)
    # print(output.shape)

    from thop import profile
    input = torch.randn(1, 4, 512, 512)
    flops, params = profile(net, inputs=(input,))
    # 9046928    904W参数量
    print(flops, params)
