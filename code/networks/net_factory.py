from networks.unet import UNet, UNet_2d, teeth, teeth_D,UNet_sdf, UNet_2d_changedropout
import torch.nn as nn

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0):
    if net_type == "unet" and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "unet_sdf" and mode == "train":
        net = UNet_sdf(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "teeth" and mode == "train":
        net = teeth(in_chns=in_chns, class_teeth_num=class_num, class_background_num = 2).cuda()
    if net_type == "teeth_D" and mode == "train":
        net = teeth_D(in_chns=in_chns, class_teeth_num=class_num, class_background_num = 2).cuda()
    return net

def BCP_net(in_chns=1, class_num=4, ema=False):
    net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
                param.requires_grad = False  # 冻结所有参数
    return net

def BCP_net_changedropout(in_chns=1, class_num=4, ema=False, weights=[0.0, 0.0, 0.5, 0.5]):
    net = UNet_2d_changedropout(in_chns=in_chns, class_num=class_num, weights=weights).cuda()
    if ema:
        for param in net.parameters():
                param.requires_grad = False  # 冻结所有参数
    return net

def BCP_net_sdf(in_chns=1, class_num=4, ema=False):
    net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
                param.requires_grad = False  # 冻结所有参数
        for param in net.decoder_sdf.parameters():
                param.requires_grad = True  # 解冻 self.decoder_sdf 的参数
    return net