import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class simple_conv(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride=(1, 1)):
        super(simple_conv, self).__init__()
        if isinstance(kernel_size, tuple):
            padding0 = int(kernel_size[0] / 2)
            padding1 = int(kernel_size[1] / 2)
            padding = (padding0, padding1)
        elif isinstance(kernel_size, int):
            padding = int(kernel_size / 2)
        else:
            raise Exception('unknow kernel_size type')
        self.conv = nn.Conv2d(in_channels=in_plane, out_channels=out_plane, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class basic_block(nn.Module):
    expansion = 1

    def __init__(self, in_plane, out_plane, stride):
        super(basic_block, self).__init__()
        self.residual = nn.Sequential(
            simple_conv(in_plane, out_plane, kernel_size=(3, 3), stride=stride),
            nn.ReLU(inplace=True),
            simple_conv(out_plane, out_plane, kernel_size=(3, 3), stride=1)
        )
        if in_plane != out_plane or stride != 1:
            self.projection = simple_conv(in_plane, out_plane, kernel_size=(1, 1), stride=stride)
        else:
            self.projection = nn.Sequential()

    def forward(self, x):
        res = self.residual(x)
        prj = self.projection(x)
        out = F.relu_(res + prj)
        return out


class bottle_neck(nn.Module):
    expansion = 4

    def __init__(self, in_plane, out_plane, stride):
        super(bottle_neck, self).__init__()
        self.residual = nn.Sequential(
            simple_conv(in_plane, out_plane, kernel_size=(1, 1), stride=1),
            nn.ReLU(inplace=True),
            simple_conv(out_plane, out_plane, kernel_size=(3, 3), stride=stride),
            nn.ReLU(inplace=True),
            simple_conv(out_plane, out_plane * self.expansion, kernel_size=(1, 1), stride=1),
        )

        if in_plane != out_plane * self.expansion or stride != 1:
            self.projection = simple_conv(in_plane, out_plane * self.expansion, kernel_size=(1, 1), stride=stride)
        else:
            self.projection = nn.Sequential()

    def forward(self, x):
        res = self.residual(x)
        prj = self.projection(x)
        out = F.relu_(res + prj)
        return out


class resnet_core(nn.Module):
    def __init__(self, block, block_stacked, origin_pic_channels=3):
        super(resnet_core, self).__init__()
        self.in_plane = 64
        self.layer0 = simple_conv(origin_pic_channels, 64, kernel_size=(7, 7), stride=(2, 2))
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_stacked[0], stride=1)
        self.layer2 = self._make_layer(block, 128, block_stacked[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_stacked[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_stacked[3], stride=2)

        self.layer5 = nn.Conv2d(in_channels=512 * block.expansion, out_channels=256, kernel_size=(3, 3), stride=2,
                                padding=1)

        self.layer6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=2, padding=1)

        self.latlayer1 = nn.Conv2d(in_channels=512 * block.expansion, out_channels=256, kernel_size=1)
        self.latlayer2 = nn.Conv2d(in_channels=256 * block.expansion, out_channels=256, kernel_size=1)
        self.latlayer3 = nn.Conv2d(in_channels=128 * block.expansion, out_channels=256, kernel_size=1)

        self.toplayer1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)

    def _make_layer(self, block, out_plane, num_of_block_stacked, stride):
        strides = num_of_block_stacked * [1]
        strides[0] = stride
        layers = []
        for spec_stide in strides:
            layers.append(block(self.in_plane, out_plane, spec_stide))
            self.in_plane = out_plane * block.expansion
        return nn.Sequential(*layers)

    def _upsample(self, slave, master):
        _, _, W, H = master.size()
        return F.upsample(slave, size=(W, H), mode='bilinear', align_corners=True) + master

    def forward(self, x):
        c1 = self.layer0(x)
        c1 = F.relu_(c1)
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c6 = self.layer5(c5)
        c7 = self.layer6(F.relu_(c6))

        p7 = c7

        p6 = c6

        p5 = self.latlayer1(c5)

        p4 = self._upsample(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)

        p3 = self._upsample(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        return p3, p4, p5, p6, p7
