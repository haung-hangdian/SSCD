# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import os
import sys
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from networks.segmentanything import sam_model_registry
from networks.segmentanything.Sam_lora import LoRA_Sam

class FeatureExtractor_unet(nn.Module):
    def __init__(self, fea_dim=[16, 32, 64, 128, 256], output_dim=256) -> None:
        super().__init__()
        assert len(fea_dim)==5, 'input_dim is not correct'
        cnt = fea_dim[0]
        self.fea0 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[1]
        self.fea1 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[2]
        self.fea2 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[3]
        self.fea3 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[4]
        self.fea4 = nn.Conv2d(in_channels=cnt, out_channels=output_dim, kernel_size=1, bias=False)
        
    def forward(self, fea_list):
        feature0 = fea_list[0]
        feature1 = fea_list[1]
        feature2 = fea_list[2]
        feature3 = fea_list[3]
        feature4 = fea_list[4]
        x = self.fea0(feature0) + feature0
        x = nn.Upsample(size = feature1.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature1), dim=1)
        x = self.fea1(x) + x
        x = nn.Upsample(size = feature2.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature2), dim=1)
        x = self.fea2(x) + x
        x = nn.Upsample(size = feature3.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature3), dim=1)
        x = self.fea3(x) + x
        x = nn.Upsample(size = feature4.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature4), dim=1)
        x = self.fea4(x) 
        # print(x.shape) # ([2, 256, 256, 256])
        return x

class FeatureExtractor_transformer(nn.Module):
    def __init__(self, fea_dim=[96, 192, 384, 768], output_dim=256) -> None:
        super().__init__()
        assert len(fea_dim)==4, 'input_dim is not correct'
        cnt = fea_dim[0]
        self.fea0 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[1]
        self.fea1 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[2]
        self.fea2 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[3]
        self.fea3 = nn.Conv2d(in_channels=cnt, out_channels=output_dim, kernel_size=1, bias=False)
        
    def forward(self, fea_list):
        feature0 = fea_list[0]
        feature1 = fea_list[1]
        feature2 = fea_list[2]
        feature3 = fea_list[3]
        x = self.fea0(feature0) + feature0
        x = nn.Upsample(size = feature1.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature1), dim=1)
        x = self.fea1(x) + x
        x = nn.Upsample(size = feature2.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature2), dim=1)
        x = self.fea2(x) + x
        x = nn.Upsample(size = feature3.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature3), dim=1)
        x = self.fea3(x)
        x = nn.Upsample(size = (14, 14), mode='bilinear', align_corners=True)(x)
        # print(x.shape) # ([2, 256, 256, 256])
        return x
    
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3*3卷积层
            nn.BatchNorm2d(out_channels),  # 对输入batch的每一个特征通道进行normalize
            nn.LeakyReLU(),  # ReLU是将所有的负值都设为零，Leaky ReLU是给所有负值赋予一个非零斜率
            nn.Dropout(dropout_p),  # Dropout层
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 3*3卷积层
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        # nn.Sequential:序列容器 nn.MaxPool2d:最大池化
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        # self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5) #ft_chns数组的长度必须是5
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        # self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv1 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        x = self.up1(x4, x3)
        x_f1 = x
        # print(x_f1.shape)
        output1 = F.interpolate(
            x_f1, 
            scale_factor=8, 
            mode='bilinear',  # 可选 'nearest' 或 'bicubic'
            align_corners=True
        )
        output1 = self.out_conv1(output1)
        # print(x.shape)
        # print(x2.shape)
        # print(output1.shape)
        x = self.up2(x, x2)
        x_f2 = x
        output2 = F.interpolate(
            x_f2, 
            scale_factor=4, 
            mode='bilinear',  # 可选 'nearest' 或 'bicubic'
            align_corners=True
        )
        output2 = self.out_conv2(output2)
        x = self.up3(x, x1)
        x_f3 = x
        output3 = F.interpolate(
            x_f3, 
            scale_factor=2, 
            mode='bilinear',  # 可选 'nearest' 或 'bicubic'
            align_corners=True
        )
        output3 = self.out_conv3(output3)
        x_f = self.up4(x, x0)
        output = self.out_conv(x_f)
        return output, x_f

class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape=(256, 256)):  ###########根据需要修改
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

class Sam_unet(nn.Module):
    def __init__(self, in_chans, num_class):
        super(Sam_unet, self).__init__()
        self.params = {'in_chns': in_chans,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': num_class,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        # self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        self.sam = sam_model_registry["vit_b"](checkpoint="/home/v1-4080s/hhy/ABD/code/networks/segmentanything/sam_vit_b_01ec64.pth")
        self.lora_sam = LoRA_Sam(self.sam,r = 4)
        self.lora_model = self.lora_sam.sam
        for name, param in self.lora_model.mask_decoder.named_parameters():
            param.requires_grad = False
        self.encoder = Encoder(self.params)
        self.img = nn.Conv2d(in_chans, 3, kernel_size=3, stride=1, padding=1)
        self.down = nn.Sequential()
        for i in range(0, 4):
            in_chns = self.ft_chns[i] + 768
            out_chns = self.ft_chns[i]
            dropout = self.dropout[i]
            self.down.add_module(
                f"down{i+1}",  # 层级名称（如down1, down2）
                ConvBlock(in_chns, out_chns, dropout)
            )
        self.decoder = Decoder(self.params)
    
    def forward(self, x):
        # _, _, H, W = x.shape
        x_1024 = self.img(x)
        x_1024 = F.interpolate(x_1024, size=(1024, 1024), mode='bicubic', align_corners=False)
        # with torch.no_grad():
        sam_feature, sam_interm = self.lora_model.image_encoder(x_1024)
        feature_unet = self.encoder(x)
        # print(feature_unet[0].shape)
        # print(feature_unet[1].shape)
        # print(feature_unet[2].shape)
        # print(feature_unet[3].shape)
        # print(feature_unet[4].shape)
        # print(len(sam_interm))
        for i in range(0, 4):
            _, _, h_embed, w_embed = feature_unet[i].shape
            # print(i)
            # print(h_embed)
            sam_interm[i+2] = F.interpolate(sam_interm[i+2], size=(h_embed, w_embed), mode='bicubic', align_corners=False)
            feature_unet[i] = self.down[i](torch.cat([sam_interm[i+2], feature_unet[i]], dim=1))
        _, _, h_embed, w_embed = feature_unet[4].shape
        sam_feature = F.interpolate(sam_feature, size=(h_embed, w_embed), mode='bicubic', align_corners=False)
        output_unet, _ = self.decoder(feature_unet)
        feature_unet[4] = sam_feature
        output_sam, _ = self.decoder(feature_unet)
        return output_unet, output_sam

class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)
        self.aux_decoder3 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg


class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg

class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, x_f = self.decoder(feature)
        return output

class UNet_numclass(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, x_f = self.decoder(feature)
        return output

class UNet_sdf(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_sdf, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.decoder_sdf = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, x_f = self.decoder(feature)
        output_sdf, x_f_sdf = self.decoder_sdf(feature)
        return output, output_sdf
    
class teeth(nn.Module):
    def __init__(self, in_chns, class_teeth_num, class_background_num):
        super(teeth, self).__init__()

        params_1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_teeth_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        
        params_2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_background_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        
        self.encoder = Encoder(params_1)
        self.decoder_teeth = Decoder(params_1)
        self.decoder_background = Decoder(params_2)

    def forward(self, x):
        feature = self.encoder(x)
        output_background, x_f_B = self.decoder_background(feature)
        output_teeth, x_f = self.decoder_teeth(feature)
        output_teeth[:, 0, :, :] = output_background[:, 0, :, :]
        return output_teeth

class teeth_D(nn.Module):
    def __init__(self, in_chns, class_teeth_num, class_background_num):
        super(teeth_D, self).__init__()

        params_1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_teeth_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        
        params_2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_background_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        
        self.encoder = Encoder(params_1)
        self.decoder_teeth = Decoder(params_1)
        self.decoder_background = Decoder(params_2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.encoder(x)
        output_background, x_f_B = self.decoder_background(feature)
        output_background = self.sigmoid(output_background) 
        weight = output_background[:, 1, :, :]
        weight = weight.unsqueeze(1)
        # print(weight.shape)
        x_teeth = x * weight
        feature_teeth = self.encoder(x_teeth)
        output_teeth, x_f = self.decoder_teeth(feature_teeth)
        return output_teeth
    
    
class UNet_2d(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_2d, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)
            
    def forward_projection_head(self, features):
        return self.projection_head(features)
        #return self.decoder(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        return output
    
def custom_weighted_sum(tensors, weights):
    """
    输入:
        tensors: 包含 5 个形状为 (B, C, H, W) 张量的列表
        weights: 权重列表，长度5，总和建议为1（如 [0.3, 0.25, 0.2, 0.15, 0.1]）
    输出:
        加权和后的张量 (B, C, H, W)
    """
    assert len(tensors) == 4, "需要5个张量"
    assert len(weights) == 4, "需要5个权重"
    assert all(t.shape == tensors[0].shape for t in tensors), "张量形状不一致"

    # 将权重转换为与张量同设备的Tensor
    weights = torch.tensor(weights, device=tensors[0].device, dtype=tensors[0].dtype)
    
    # 加权求和（自动广播）
    weighted_sum = torch.sum(
        torch.stack(tensors) * weights.view(-1, 1, 1, 1, 1),  # 权重形状调整为 [5,1,1,1,1] 以便广播
        dim=0
    )
    return weighted_sum

class UNet_2d_changedropout(nn.Module):
    def __init__(self, in_chns, class_num, weights = [0.25, 0.25, 0.25, 0.25]):
        super(UNet_2d_changedropout, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)
        dim_in = 16
        feat_dim = 32
        self.weights = weights
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)
            
    def forward_projection_head(self, features):
        return self.projection_head(features)
        #return self.decoder(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, output_feature_3, output_feature_2, output_feature_1 = self.decoder(feature)
        # print(output.shape)
        # print(output_feature_3.shape)
        # print(output_feature_2.shape)
        # print(output_feature_1.shape)
        
        tensors = [output_feature_3,output_feature_2,output_feature_1,output]
        result = custom_weighted_sum(tensors, self.weights)
        return output, result
    
class UNet_2d_sdf(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_2d_sdf, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.decoder_sdf = Decoder(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)
            
    def forward_projection_head(self, features):
        return self.projection_head(features)
        #return self.decoder(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        output_sdf, features_sdf = self.decoder_sdf(feature)
        return output, output_sdf