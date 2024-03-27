import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
from train_util import *
import pdb
import math
from functools import partial
from einops.layers.torch import Rearrange, Reduce

# define the function includes in import *
__all__ = [
    'SpixelNet1l', 'SpixelNet1l_bn'
]

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        # self.con1 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.con1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.con1(self.fn(self.norm(x)) + x)


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.PReLU(),
        dense(inner_dim, dim),
        nn.PReLU()
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, expansion_factor=4, expansion_factor_token=0.5,
             dropout=0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size,
                  h=int(image_h / patch_size), w=int(image_w / patch_size)),
    )


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1)  # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1)  # norm to [0,1] NxHxWx1
        hg, wg = hg * 2 - 1, wg * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide)
        return coeff.squeeze(2)


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,
                 use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size,
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=bn)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)

        return output


class Recurrent_Attn(nn.Module):
    def __init__(self, num_class):
        super(Recurrent_Attn, self).__init__()

        self.QConv1 = conv(True, 256, 256, 3)
        self.KConv1 = conv(True, 256, 256, 3)
        self.VConv1 = conv(True, 256, 256, 3)

        kernel_size = 3
        stride = 1
        self.num_class = num_class
        self.QConv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=stride, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        self.KConv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        self.VConv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

    def self_attn(self, Q, K, V):
        # projection the Q, K and V
        b, c, h, w = Q.shape
        # B x C x H x W ==> B X C x 9 x H x W
        K_unfold = F.unfold(K, (3, 3), padding=1)
        K_unfold = K_unfold.view(b, c, 9, h, w)

        # B x C x H x W ==> B X C x 9 x H x W
        V_unfold = F.unfold(V, (3, 3), padding=1)
        V_unfold = V_unfold.view(b, c, 9, h, w)

        Q_unfold = Q.unsqueeze(2)

        # dot = torch.exp(Q_unfold * K_unfold / math.sqrt(c*1.))
        # dot = torch.sum(Q_unfold * K_unfold, dim=1, keepdim=True) / math.sqrt(c*1.)
        dot = Q_unfold * K_unfold / math.sqrt(c * 1.)
        dot = F.softmax(dot, dim=2)

        attn = torch.sum(dot * V_unfold, dim=2)

        return attn

    def forward(self, x):
        Q1 = self.QConv1(x)
        K1 = self.KConv1(x)
        V1 = self.VConv1(x)

        attn1 = self.self_attn(Q1, K1, V1)
        # attn1 = attn1 + x

        Q2 = self.QConv2(attn1)
        K2 = self.KConv2(attn1)
        V2 = self.VConv2(attn1)

        attn2 = self.self_attn(Q2, K2, V2)

        return attn2


class SpixelNet(nn.Module):
    expansion = 1

    def __init__(self, dataset='', batchNorm=True, Train=False):
        super(SpixelNet, self).__init__()
        self.Train = Train
        self.batchNorm = batchNorm
        self.assign_ch = 9
        input_chs = 3
        self.conv0a = conv(self.batchNorm, input_chs, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)
        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)
        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)
        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)
        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)
        self.dhac = DHACblock(256)
        self.deconv3 = deconv(256, 128)
        self.cbam3 = cbam(256)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.deconv2 = deconv(128, 64)
        self.cbam2 = cbam(128)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.deconv1 = deconv(64, 32)
        self.cbam1 = cbam(64)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.deconv0 = deconv(32, 16)
        self.cbam0 = cbam(32)
        self.conv0_1 = conv(self.batchNorm, 32, 16)
        self.pred_mask0 = predict_mask(16, self.assign_ch)
        self.softmax = nn.Softmax(1)
        mask_select = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).reshape(3, 3)
        mask_select = mask_select.repeat(208, 208)
        self.mask_select = mask_select.view(1, 1, 208 * 3, 208 * 3).float().cuda()
        self.sp_pred = Recurrent_Attn(num_class=50)
        self.bridge_sp1 = conv(self.batchNorm, 256, 64, kernel_size=3)
        self.bridge_sp2 = conv(self.batchNorm, 64, 16, kernel_size=3)
        self.merge_sp = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
        )
        self.g = GuideNN()
        self.realCA = ChannelAttentionMLP(3, ratio=1)
        self.imagCA = ChannelAttentionMLP(3, ratio=1)
        self.f1 = MLPMixer(
            image_size=128,
            channels=3,
            patch_size=16,
            dim=768,
            depth=12
        )
        self.f2 = MLPMixer(
            image_size=128,
            channels=3,
            patch_size=16,
            dim=768,
            depth=12
        )
        self.pred_mask3 = predict_mask(128, self.assign_ch)
        self.pred_mask2 = predict_mask(64, self.assign_ch)
        self.pred_mask1 = predict_mask(32, self.assign_ch)
        self.squeeze_3d = nn.Conv3d(24, 3, kernel_size=3, stride=1, padding=1)
        self.slice = Slice()
        self.fusion_conv1 = nn.Conv2d(19, 19, 3, stride=1, padding=1)
        self.fusion_conv2 = nn.Conv2d(19, 16, 1, stride=1)
        self.pReLU = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def expand_sp(self, sp_feat):
        sp_feat = F.pad(sp_feat, [1, 1, 1, 1])
        b, c, h, w = sp_feat.shape
        output_list = []
        # this loop is acceptable due to the lower h, w
        for i in range(1, h - 1):
            row_list = []
            for j in range(1, w - 1):
                sp_patch = sp_feat[:, :, (i - 1):(i + 2), (j - 1):(j + 2)]
                sp_patch = sp_patch.repeat(1, 1, 16, 16)
                row_list.append(sp_patch)

            output_list.append(torch.cat(row_list, dim=-1))

        output = torch.cat(output_list, dim=-2)

        return output

    def expand_pixel(self, pixel_feat):
        b, c, h, w = pixel_feat.shape
        pixel_feat = pixel_feat.view(b, c, h, 1, w, 1)
        pixel_feat = pixel_feat.repeat(1, 1, 1, 3, 1, 3)
        pixel_feat = pixel_feat.reshape(b, c, h * 3, w * 3)

        pixel_feat = pixel_feat * self.mask_select

        return pixel_feat

    def forward(self, x, patch_posi=None, patch_label=None):
        if not self.Train:
            mask_select = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).reshape(3, 3)
            b, c, h, w = x.shape
            mask_select = mask_select.repeat(h, w)
            self.mask_select = mask_select.view(1, 1, h * 3, w * 3).float().cuda()
        out1 = self.conv0b(self.conv0a(x))
        out2 = self.conv1b(self.conv1a(out1))
        out3 = self.conv2b(self.conv2a(out2))
        out4 = self.conv3b(self.conv3a(out3))
        out5 = self.conv4b(self.conv4a(out4))
        out5 = self.dhac(out5)
        out5_attn = self.sp_pred(out5)
        out5 = out5 + out5_attn
        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        concat3 = concat3 + self.cbam3(concat3)
        out_conv3_1 = self.conv3_1(concat3)
        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        concat2 = concat2 + self.cbam2(concat2)
        out_conv2_1 = self.conv2_1(concat2)
        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        concat1 = concat1 + self.cbam1(concat1)
        out_conv1_1 = self.conv1_1(concat1)
        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        concat0 = concat0 + self.cbam0(concat0)
        out_conv0_1 = self.conv0_1(concat0)
        sp_map = self.bridge_sp2(self.bridge_sp1(out5))
        sp_expand = self.expand_sp(sp_map)
        pixel_expand = self.expand_pixel(out_conv0_1)
        merged = sp_expand + pixel_expand
        out = self.merge_sp(merged)
        x_f = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)
        real = x_f[..., 0]
        imag = x_f[..., 1]
        real = real + self.realCA(real)
        imag = imag + self.imagCA(imag)
        feature_real = F.interpolate(real, (128, 128),
                                     mode='bilinear', align_corners=True)
        feature_imag = F.interpolate(imag, (128, 128),
                                     mode='bilinear', align_corners=True)
        x1 = self.f1(feature_real).view(-1, 12, 16, 16, 16)
        x2 = self.f2(feature_imag).view(-1, 12, 16, 16, 16)
        coeff = self.squeeze_3d(torch.cat((x1, x2), dim=1))
        g = self.g(x)
        feat_frequency = self.slice(coeff, g)
        out = self.pReLU(self.fusion_conv2(self.pReLU(self.fusion_conv1(torch.cat((feat_frequency, out), dim=1)))))
        mask0 = self.pred_mask0(out)
        prob0 = self.softmax(mask0)
        return prob0, out_conv0_1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def SpixelNet1l(data=None):
    # Model without  batch normalization
    model = SpixelNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def SpixelNet1l_bn(dataset='BDS500', data=None, Train=False):
    # model with batch normalization
    model = SpixelNet(dataset=dataset, batchNorm=True, Train=Train)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model
