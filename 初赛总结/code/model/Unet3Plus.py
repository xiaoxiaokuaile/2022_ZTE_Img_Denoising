# Learner: 王振强
# Learn Time: 2022/4/12 13:07
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


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


# 上采样下采样以及通道数调整
class Unet_up_down(nn.Module):
    def __init__(self, in_chanel, out_chanel, up_num, mode='down'):
        super(Unet_up_down, self).__init__()
        # 上采样
        if mode == 'down':
            self.conv = nn.Sequential(
                nn.MaxPool2d(up_num, up_num, ceil_mode=True),
                nn.Conv2d(in_chanel, out_chanel, 3, padding=1),
                nn.BatchNorm2d(out_chanel),
                nn.LeakyReLU(negative_slope=0.125),
            )
        elif mode == 'level':
            self.conv = nn.Sequential(
                nn.Conv2d(in_chanel, out_chanel, 3, padding=1),
                nn.BatchNorm2d(out_chanel),
                nn.LeakyReLU(negative_slope=0.125),
            )
        elif mode == 'up':
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=up_num, mode='bilinear'),
                nn.Conv2d(in_chanel, out_chanel, 3, padding=1),
                nn.BatchNorm2d(out_chanel),
                nn.LeakyReLU(negative_slope=0.125),
            )
        else:
            print("Input Error!!!")

    def forward(self, input):
        output = self.conv(input)
        return output


# UNet 3+ 返回解码过程中的每一次输出
class UNet3Plus(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, is_batchnorm=False, out_num=False):
        super(UNet3Plus, self).__init__()
        self.is_batchnorm = is_batchnorm
        # 返回5个输出或最后一个输出,默认一个输出
        self.out_num = out_num
        filters = [32, 64, 128, 256, 512]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        self.h1_hd4 = Unet_up_down(filters[0], self.CatChannels, 8, mode='down')
        self.h2_hd4 = Unet_up_down(filters[1], self.CatChannels, 4, mode='down')
        self.h3_hd4 = Unet_up_down(filters[2], self.CatChannels, 2, mode='down')
        self.h4_hd4 = Unet_up_down(filters[3], self.CatChannels, 1, mode='level')
        self.h5_hd4 = Unet_up_down(filters[4], self.CatChannels, 2, mode='up')
        self.conv_hd4 = Unet_up_down(self.UpChannels, self.UpChannels, 1, mode='level')

        '''stage 3d'''
        self.h1_hd3 = Unet_up_down(filters[0], self.CatChannels, 4, mode='down')
        self.h2_hd3 = Unet_up_down(filters[1], self.CatChannels, 2, mode='down')
        self.h3_hd3 = Unet_up_down(filters[2], self.CatChannels, 1, mode='level')
        self.h4_hd3 = Unet_up_down(filters[3], self.CatChannels, 2, mode='up')
        self.h5_hd3 = Unet_up_down(filters[4], self.CatChannels, 4, mode='up')
        self.conv_hd3 = Unet_up_down(self.UpChannels, self.UpChannels, 1, mode='level')

        '''stage 2d '''
        self.h1_hd2 = Unet_up_down(filters[0], self.CatChannels, 2, mode='down')
        self.h2_hd2 = Unet_up_down(filters[1], self.CatChannels, 1, mode='level')
        self.h3_hd2 = Unet_up_down(filters[2], self.CatChannels, 2, mode='up')
        self.h4_hd2 = Unet_up_down(filters[3], self.CatChannels, 4, mode='up')
        self.h5_hd2 = Unet_up_down(filters[4], self.CatChannels, 8, mode='up')
        self.conv_hd2 = Unet_up_down(self.UpChannels, self.UpChannels, 1, mode='level')

        '''stage 1d'''
        self.h1_hd1 = Unet_up_down(filters[0], self.CatChannels, 1, mode='level')
        self.h2_hd1 = Unet_up_down(filters[1], self.CatChannels, 2, mode='up')
        self.h3_hd1 = Unet_up_down(filters[2], self.CatChannels, 4, mode='up')
        self.h4_hd1 = Unet_up_down(filters[3], self.CatChannels, 8, mode='up')
        self.h5_hd1 = Unet_up_down(filters[4], self.CatChannels, 16, mode='up')
        self.conv_hd1 = Unet_up_down(self.UpChannels, self.UpChannels, 1, mode='level')

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, out_channels, (3,3), padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, out_channels, (3,3), padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, out_channels, (3,3), padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, out_channels, (3,3), padding=1)
        self.outconv5 = nn.Conv2d(filters[4], out_channels, (3,3), padding=1)

    def forward(self, inputs):
        n, c, h, w = inputs.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_image = F.pad(inputs, (0, w_pad, 0, h_pad), 'replicate')
        ## -------------Encoder-------------
        h1 = self.conv1(padded_image)
        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)
        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)
        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)
        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_hd4(h1)    # torch.Size([1, 32, 64, 64])
        h2_PT_hd4 = self.h2_hd4(h2)    # torch.Size([1, 32, 64, 64])
        h3_PT_hd4 = self.h3_hd4(h3)    # torch.Size([1, 32, 64, 64])
        h4_Cat_hd4 = self.h4_hd4(h4)   # torch.Size([1, 32, 64, 64])
        hd5_UT_hd4 = self.h5_hd4(hd5)  # torch.Size([1, 32, 64, 64])
        hd4 = self.conv_hd4(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))

        h1_PT_hd3 = self.h1_hd3(h1)    # torch.Size([1, 32, 64, 64])
        h2_PT_hd3 = self.h2_hd3(h2)    # torch.Size([1, 32, 64, 64])
        h3_Cat_hd3 = self.h3_hd3(h3)    # torch.Size([1, 32, 64, 64])
        hd4_UT_hd3 = self.h4_hd3(h4)   # torch.Size([1, 32, 64, 64])
        hd5_UT_hd3 = self.h5_hd3(hd5)  # torch.Size([1, 32, 64, 64])
        hd3 = self.conv_hd3(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))

        h1_PT_hd2 = self.h1_hd2(h1)    # torch.Size([1, 32, 64, 64])
        h2_Cat_hd2 = self.h2_hd2(h2)    # torch.Size([1, 32, 64, 64])
        hd3_UT_hd2 = self.h3_hd2(h3)    # torch.Size([1, 32, 64, 64])
        hd4_UT_hd2 = self.h4_hd2(h4)   # torch.Size([1, 32, 64, 64])
        hd5_UT_hd2 = self.h5_hd2(hd5)  # torch.Size([1, 32, 64, 64])
        hd2 = self.conv_hd2(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))

        h1_Cat_hd1 = self.h1_hd1(h1)    # torch.Size([1, 32, 64, 64])
        hd2_UT_hd1 = self.h2_hd1(h2)    # torch.Size([1, 32, 64, 64])
        hd3_UT_hd1 = self.h3_hd1(h3)    # torch.Size([1, 32, 64, 64])
        hd4_UT_hd1 = self.h4_hd1(h4)   # torch.Size([1, 32, 64, 64])
        hd5_UT_hd1 = self.h5_hd1(hd5)  # torch.Size([1, 32, 64, 64])
        hd1 = self.conv_hd1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)
        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)
        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)
        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)
        d1 = self.outconv1(hd1)

        d1 = d1[:, :, :h, :w]
        d2 = d2[:, :, :h, :w]
        d3 = d3[:, :, :h, :w]
        d4 = d4[:, :, :h, :w]
        d5 = d5[:, :, :h, :w]

        if self.out_num:
            return d1
        else:
            return d1, d2, d3, d4, d5

    def save(self, circle):
        name = "./weights/UNet3Plus" + str(circle) + ".pth"
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
    net = UNet3Plus()
    input = torch.from_numpy(np.random.randn(1, 4, 512, 512)).float()
    output1,output2,output3,output4,output5 = net(input)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
    print(output4.shape)
    print(output5.shape)
    #
    # from thop import profile
    # input = torch.randn(1, 4, 512, 512)
    # flops, params = profile(net, inputs=(input,))
    # # 861165465600  6749636    2600W参数量
    # print(flops, params)