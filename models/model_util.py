import torch
import torch.nn as nn
from reprlib import recursive_repr
import torch.nn.functional as F
from functools import partial


## *************************** my functions ****************************

def predict_param(in_planes, channel=3):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)


def predict_mask(in_planes, channel=9):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)


def predict_feat(in_planes, channel=20, stride=1):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=stride, padding=1, bias=True)


def predict_prob(in_planes, channel=9):
    return nn.Sequential(
        nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Softmax(1)
    )


# ***********************************************************************

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class cbam(nn.Module):
    def __init__(self, channels):
        super(cbam, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


nonlinearity = partial(F.relu, inplace=True)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, size=size, mode='bilinear', align_corners=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttentionMLP(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionMLP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avgout = self.sharedMLP(self.avg_pool(x).view(b, c))
        maxout = self.sharedMLP(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avgout + maxout).view(b, c, 1, 1)


class DHACblock(nn.Module):
    def __init__(self, channel):
        super(DHACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate6 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate7 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate8 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate9 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.gap = ASPPPooling(256, 256)
        self.conv_bn_relu = conv(True, 1280, 256, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        gap = self.gap(x)
        dilate1_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        dilate2_out = nonlinearity(
            self.conv1x1(self.dilate6(self.dilate5(self.dilate4(self.dilate3(self.dilate2(self.dilate1(x))))))))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate9(self.dilate8(
            self.dilate7(self.dilate6(self.dilate5(self.dilate4(self.dilate3(self.dilate2(self.dilate1(x)))))))))))
        return self.conv_bn_relu(torch.cat((x, gap, dilate1_out, dilate2_out, dilate3_out), dim=1))
