import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile

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


class Unet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(Unet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
        )

        self.upv6 = nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=(2,2))
        self.conv_6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
        )

        self.upv7 = nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=(2,2))
        self.conv_7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
        )

        self.upv8 = nn.ConvTranspose2d(128, 64, kernel_size=(2,2), stride=(2,2))
        self.conv_8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
        )

        self.upv9 = nn.ConvTranspose2d(64, 32, kernel_size=(2,2), stride=(2,2))
        self.conv_9 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.LeakyReLU(negative_slope=0.125),
        )

        self.conv_10 = nn.Conv2d(32, out_channels, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        n, c, h, w = x.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_image = F.pad(x, (0, w_pad, 0, h_pad), 'replicate')

        # down
        conv1 = self.conv_1(padded_image)
        pool1 = self.pool1(conv1)

        conv2 = self.conv_2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv_3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv_4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv_5(pool4)

        # up
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv_6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv_7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv_8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv_9(up9)

        conv10 = self.conv_10(conv9)
        out = conv10[:, :, :h, :w]

        return out

    def save(self, circle):
        name = "./weights/unet" + str(circle) + ".pth"
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
    test_input = torch.from_numpy(np.random.randn(1, 4, 512, 512)).float()
    net = Unet()
    output = net(test_input)
    print(output.shape)
    # print(net)

    input = torch.randn(1, 4, 512, 512)
    flops, params = profile(net, inputs=(input,))
    # 861165465600  7760484    776W参数量
    print(flops, params)

