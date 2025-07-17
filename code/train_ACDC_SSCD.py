import argparse
import logging
import os
import random
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.distributions import Categorical
from scipy.ndimage import distance_transform_edt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from utils.displacement import ABD_I_BCP, ABD_I_Conf,ABD_R_BS, BD_S_Unlabel, Ncut, Overlap, layer, hhh
from dataloaders.dataset import (BaseDataSets, TwoStreamBatchSampler, WeakStrongAugment, RandomGenerator)
from networks.net_factory import BCP_net, BCP_net_sdf, BCP_net_changedropout
from networks.config import get_config
from networks.vision_transformer import SwinAgentUnet as ViT_seg
from utils import ramps, losses
from val_2D import test_single_volume, test_single_volume_result

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/v1-4080s/hhy/ABD/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='layer_weight1_half2', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=80000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--image_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')
# patch size
parser.add_argument('--patch_size', type=int, default=32, help='patch_size')
parser.add_argument('--h_size', type=int, default=8, help='h_size')
parser.add_argument('--w_size', type=int, default=8, help='w_size')
# top num
parser.add_argument('--top_num', type=int, default=4, help='top_num')
parser.add_argument('--cfg', type=str,
                    default="/home/v1-4080s/hhy/ABD/code/configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+', )
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
args = parser.parse_args()
config = get_config(args)

dice_loss = losses.DiceLoss(n_classes=4)

def load_net(net, path, verbose=True):
    """
    选择性加载模型参数（仅加载名称匹配的参数）
    参数：
        net: 要加载参数的模型
        path: 预训练权重路径
        verbose: 是否打印加载信息
    """
    pretrained_dict = torch.load(str(path))['net']
    model_dict = net.state_dict()
    
    # 1. 过滤匹配参数（名称和形状一致）
    matched_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    # 2. 更新模型参数
    model_dict.update(matched_dict)
    net.load_state_dict(model_dict)
    
    # 3. 打印调试信息
    if verbose:
        matched_keys = set(matched_dict.keys())
        pretrained_keys = set(pretrained_dict.keys())
        logging.info(f"成功加载参数: {matched_keys & pretrained_keys}")
        logging.warning(f"形状不匹配被忽略: {pretrained_keys - matched_keys}")
        logging.warning(f"新模型特有参数未加载: {set(model_dict) - set(pretrained_dict)}")

def load_net_opt(net, optimizer, path, verbose=True):
    # 加载模型参数
    load_net(net, path, verbose)
    
    # 加载优化器状态
    checkpoint = torch.load(str(path))
    if 'opt' not in checkpoint:
        logging.warning("检查点无优化器状态")
        return
    
    pretrained_opt = checkpoint['opt']
    current_opt = optimizer.state_dict()
    
    # 关键修复：保留当前优化器的参数组结构
    filtered_state_dict = {
        'state': {},
        'param_groups': current_opt['param_groups']  # 使用当前参数组结构
    }
    
    # 构建参数ID到模型参数的映射
    model_params = {id(p): p for _, p in net.named_parameters()}
    
    # 过滤可加载的优化器状态
    for param_id, state in pretrained_opt['state'].items():
        # 仅当参数存在于当前模型时加载状态
        if param_id in model_params:
            filtered_state_dict['state'][param_id] = state
    
    # 加载过滤后的状态
    optimizer.load_state_dict(filtered_state_dict)
    
    if verbose:
        loaded = len(filtered_state_dict['state'])
        total = len(pretrained_opt['state'])
        logging.info(f"成功加载优化器状态: {loaded}/{total}")




def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    

def generate_masks(out_believe, out_entropy, batch_size, k=0.2):
    """
    生成高置信度和低熵的掩码
    Args:
        out_believe: 置信度张量，shape [B, H, W]
        out_entropy: 熵张量，shape [B, H, W]
        batch_size: 批量大小
        k: 前k比例 (默认取前50%)
    Returns:
        mask_confidence: 高置信度掩码，shape [B, H, W]
        mask_entropy: 低熵掩码，shape [B, H, W]
    """
    with torch.no_grad():
        # 计算置信度掩码
        values_conf = out_believe.view(batch_size, -1)  # [B, H*W]
        k_val = int(k * values_conf.size(1))  # 前k的像素数量
        values, _ = values_conf.kthvalue(k_val, dim=1)  # values.shape = [B]
        threshold_conf = values  # 直接使用
        mask_confidence = (out_believe >= threshold_conf.view(-1, 1, 1)).float()

        # 计算熵掩码
        values_ent = out_entropy.view(batch_size, -1)  # 形状 [B, C*H*W]
        _, height, width = out_entropy.size()
        total_pixels = height * width  # 总像素数
        k_val = int(k * total_pixels)  # 前30%对应的像素数量
        values_ent, _ = values_ent.kthvalue(total_pixels - k_val, dim=1)  # values_ent.shape = [B]
        threshold_ent = values_ent  # 直接使用
        mask_entropy = (out_entropy >= threshold_ent.view(-1, 1, 1)).float()

    return mask_confidence, mask_entropy

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    probs_select = pseudo_variance_gate(output)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs_select)   
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Updates the EMA model parameters using the parameters from the main model.
    
    Args:
        model: The main model.
        ema_model: The EMA model.
        alpha: The EMA decay rate.
        global_step: The current training step, used to adjust EMA initialization.
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    
    # Freeze all parameters in ema_model except decoder_sdf
    for name, ema_param in ema_model.named_parameters():
        if "decoder_sdf" not in name:  # Freeze all except decoder_sdf
            ema_param.requires_grad = False

    # Update EMA for unfrozen parameters
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if ema_param.requires_grad:  # Update only if the parameter is unfrozen
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()

def pseudo_variance_gate(pred_logits, TPV=0.80, TS=0.3):
    """
    基于伪方差的门控伪标签生成
    :param pred_logits: 模型输出的logits张量 (batch_size, num_classes, H, W)
    :param TPV: 伪方差阈值，控制是否选择top-2置信度
    :param TS: top-2置信度阈值，过滤低置信预测
    :return: 伪标签 (batch_size, H, W)
    """
    # 转换为概率分布
    probs = torch.softmax(pred_logits, dim=1)  # (B, C, H, W)
    batch_size, num_classes, H, W = probs.shape
    
    # Step 1: 获取top-1和top-2的置信度及索引
    top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)  # (B, 2, H, W)
    top1_probs = top2_probs[:, 0]  # (B, H, W)
    top2_probs = top2_probs[:, 1]  # (B, H, W)
    top1_indices = top2_indices[:, 0]  # (B, H, W)
    top2_indices = top2_indices[:, 1]  # (B, H, W)

    # Step 2: 计算每个像素的伪方差
    # 公式: PV = (n-1)/n * Σ[(x_i - (1-x_i)/(n-1))^2]
    n = num_classes
    x_sum = 1.0 - probs  # (B, C, H, W)
    x_minus_i_mean = x_sum / (n - 1 + 1e-8)  # 避免除零
    squared_diff = (probs - x_minus_i_mean) ** 2  # (B, C, H, W)
    pv = ( (n-1)/n ) * torch.sum(squared_diff, dim=1)  # (B, H, W)

    # Step 3: 门控逻辑
    # 条件1: PV > TPV 时选择top-2
    # 条件2: top-2置信度 > TS 时生效
    condition = (pv > TPV) & (top2_probs > TS)
    pseudo_labels = torch.where(
        condition,
        top2_indices,  # 满足条件时选top-2类别
        top1_indices    # 否则选top-1类别
    )

    return pseudo_labels

def entropy_from_softmax(prob_tensor):
    """
    计算softmax后的信息熵。
    
    参数:
        prob_tensor (torch.Tensor): 形状为 (B, C, H, W) 的张量，已经经过Softmax处理。
    
    返回:
        torch.Tensor: 形状为 (B, H, W) 的信息熵图。
    """
    # 计算信息熵: H(X) = -sum(p * log(p))
    entropy = -torch.sum(prob_tensor * torch.log(prob_tensor.clamp(min=1e-8)), dim=1)  # 避免log(0)问题
    return entropy  # 输出形状为 (B, H, W)

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def pre_train(args, snapshot_path):
    ce_loss = CrossEntropyLoss()
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model = BCP_net(in_chns=1, class_num=num_classes)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices//2))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    # trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
    #                          num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print(sdf_batch.shape)

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            # outputs = model(volume_batch[:args.labeled_bs])
            # outputs_soft = torch.softmax(outputs, dim=1)

            # loss_ce = ce_loss(outputs, label_batch[:args.labeled_bs].long())
            # loss_dice = dice_loss(outputs_soft, label_batch[:args.labeled_bs].unsqueeze(1))
            # loss = 0.5 * (loss_dice + loss_ce)

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            # lab_a_sdf, lab_b_sdf = sdf_batch[:labeled_sub_bs], sdf_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)
            # gt_mixl_sdf = lab_a_sdf * img_mask + lab_b_sdf * (1 - img_mask)

            #-- original
            net_input_1 = img_a * img_mask + img_b * (1 - img_mask)
            batch_size, _, _, _ = net_input_1.shape
            # # out, out_sdf = model(volume_batch)
            out_mixl_1= model(net_input_1)
            
            # # loss_sdf = mix_loss_sdf(out_mixl_sdf, lab_a_sdf, lab_b_sdf, loss_mask, u_weight=1.0, unlab=True)
            # loss = (loss_dice + loss_ce ) / 2    

            out= model(volume_batch)
            out_mixl_1_soft = torch.softmax(out_mixl_1, dim=1)
            out_believe = torch.max(out_mixl_1_soft.detach(), dim=1)[0]
            out_entropy = entropy_from_softmax(out_mixl_1_soft)
            # mask_1 = (out_believe > 0.8).float()  # 注意：直接使用 float() 替代 torch.where
            # mask_2 = (out_entropy < 2).float()
            
            

            # 合并掩码并扩展维度
            mask_confidence, mask_entropy = generate_masks(out_believe, out_entropy, batch_size, k = 0.2)
            combined_mask = (mask_confidence * mask_entropy).bool()
            
            net_input = net_input_1 * mask_entropy.unsqueeze(1)
            out_mixl= model(net_input)
            out_mixl_soft = torch.softmax(out_mixl, dim=1)
            # loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            # loss_sdf = mix_loss_sdf(out_mixl_sdf, lab_a_sdf, lab_b_sdf, loss_mask, u_weight=1.0, unlab=True)
            # loss = (loss_dice + loss_ce ) / 2  
            # # 应用掩码到预测值和标签
            # masked_out = out * combined_mask.unsqueeze(1)  # [B, C, H, W]
            masked_labels = gt_mixl * mask_entropy  # [B, 1, H, W]

            # # 损失计算（仅掩码区域有效）
            # loss_dice_2 = dice_loss(out_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1).float())
            # loss_ce_2 = ce_loss(out[:args.labeled_bs], label_batch[:args.labeled_bs].long())
            loss_dice_1, loss_ce_1 = mix_loss(out_mixl_1, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            # 掩码区域的自定义损失（需显式传递掩码）
            if iter_num > 6000:
                loss_ce_en = ce_loss(
                    out_mixl, 
                    masked_labels.long()
                )
                loss_dice_en = dice_loss(
                    out_mixl_soft, 
                    masked_labels.unsqueeze(1).float(),
                    mask=mask_entropy.unsqueeze(1)  # 显式传递掩码
                )
            else:
                loss_ce_en = 0
                loss_dice_en = 0
            loss_dice = loss_dice_1 + loss_dice_en
            loss_ce = loss_ce_1 + loss_ce_en
            # loss_dice = loss_dice_1
            # loss_ce = loss_ce_1 
            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)  
            # writer.add_scalar('info/sdf_loss', loss_sdf, iter_num)   

            # logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            # if iter_num % 20 == 0:
            #     image = net_input[1, 0:1, :, :]
            #     writer.add_image('pre_train/Mixed_Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
            #     writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = gt_mixl[1, ...].unsqueeze(0) * 50
            #     writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_first_train(args ,pre_snapshot_path, snapshot_path):

    def get_comp_loss(weak, strong):
        """get complementary loss and adaptive sample weight.
        Compares least likely prediction (from strong augment) with argmin of weak augment.

        Args:
            weak (batch): weakly augmented batch
            strong (batch): strongly augmented batch

        Returns:
            comp_loss, as_weight
        """
        ce_loss = CrossEntropyLoss()
        il_output = torch.reshape(
            strong,
            (
                args.batch_size,
                args.num_classes,
                args.image_size[0] * args.image_size[1],
            ),
        )
        # calculate entropy for image-level preds (tensor of length labeled_bs)
        as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.image_size[0] * args.image_size[1]))
        # batch level average of entropy
        as_weight = torch.mean(as_weight)
        # complementary loss
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        comp_loss = as_weight * ce_loss(
            torch.add(torch.negative(strong), 1),
            comp_labels,
        )
        return comp_loss, as_weight

    def normalize(tensor):
            min_val = tensor.min(1, keepdim=True)[0]
            max_val = tensor.max(1, keepdim=True)[0]
            result = tensor - min_val
            result = result / max_val
            return result
        
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model_1 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[1.0, 0.0, 0.0, 0.0])
    model_2 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[1.0, 0.0, 0.0, 0.0])
    # model_2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Train labeled {} samples".format(labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices//2))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer_ema = optim.SGD(ema_model.decoder_sdf.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net(ema_model, pre_trained_model)
    load_net_opt(model_1, optimizer1, pre_trained_model)
    load_net_opt(model_2, optimizer2, pre_trained_model)
    # model_2.load_from(config)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_1.train()
    model_2.train()
    ema_model.train()

    iter_num = 0
    model_1_stop = 0
    model_2_stop = 0
    p_small = [64, 32, 16]
    p2 = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    min_entropy = 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # print(volume_batch.shape)
            # print(label_batch.shape)
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]


            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b = get_ACDC_masks(pre_b, nms=1)

                pre_a_s = ema_model(uimg_a_s)
                pre_b_s = ema_model(uimg_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)


                img_mask, loss_mask = generate_mask(img_a)
                # unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                # l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)


            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0) 

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask) 
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            # Model1 Loss
            out_unl_1, out_unl_1_result = model_1(net_input_unl_1)
            out_l_1 , out_l_1_result= model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            B1, _, _, _ = out_1.size()
            B2, _, _, _ = net_input_unl_1.size()
            out_soft_1 = torch.softmax(out_1, dim=1)
            C = out_soft_1.size(1)  # 获取类别数
            normalization = torch.log(torch.tensor(C, device=out_soft_1.device))  # 理论最大熵
            mean_entropy_1 = -torch.mean(
                torch.sum(out_soft_1 * torch.log(out_soft_1 + 1e-10), dim=1)
            ) / normalization
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            out_top2_soft_1 = pseudo_variance_gate(out_1)
            out_1_entropy = entropy_from_softmax(out_soft_1)
            out_1_result = torch.cat([out_unl_1_result, out_l_1_result], dim=0)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_label_1 = torch.max(out_soft_1.detach(), dim=1)[1]
            
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2, out_unl_2_result = model_2(net_input_unl_2)
            out_l_2, out_l_2_result = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_top2_soft_2 = pseudo_variance_gate(out_2)
            out_soft_2 = torch.softmax(out_2, dim=1)
            C = out_soft_2.size(1)  
            normalization = torch.log(torch.tensor(C, device=out_soft_2.device))
            mean_entropy_2 = -torch.mean(
                torch.sum(out_soft_2 * torch.log(out_soft_2 + 1e-10), dim=1)
            ) / normalization
            out_2_result = torch.cat([out_unl_2_result, out_l_2_result], dim=0)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_label_2 = torch.max(out_soft_2.detach(), dim=1)[1]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False) 
            out_2_entropy = entropy_from_softmax(out_soft_2)
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            mean_entropy = (mean_entropy_1 + mean_entropy_2) / 2
            if mean_entropy < min_entropy :
                min_entropy = mean_entropy

            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))  
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1)) 
            # pseudo_supervision1_top2 = dice_loss(out_soft_1, out_2_label_init.unsqueeze(1).float(), mask=mask_3.unsqueeze(1))  
            # pseudo_supervision2_top2 = dice_loss(out_soft_2, out_1_label_init.unsqueeze(1).float(), mask=mask_3_2.unsqueeze(1))  
            # if model_1_stop == 100 or model_2_stop == 100:
            #     p2 = (p2 + 1) % 3
            p = iter_num/max_iterations

            image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1_result, out_2_result,args)
            # image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)

            # print(out_max_1.shape)
            # image_patch_last = Ncut(out_max_1, out_max_2, net_input_1, net_input_2, p, args)
            B3, _, _ = image_patch_last.size()
            image_output_1,_ = model_1(image_patch_last.unsqueeze(1))  
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            # pseduo_mask_1 = (normalize(image_output_1) > (1-p**2)).float()
            # pseduo_masknew_1 = pseduo_mask_1 * image_output_1
            image_output_max_1 = torch.max(image_output_soft_1.detach(), dim=1)[0]
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_out_top2_soft_1 = pseudo_variance_gate(image_output_1)
            image_out_1_entropy = entropy_from_softmax(image_output_soft_1)
            image_output_2,_ = model_2(image_patch_last.unsqueeze(1))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            image_output_max_2 = torch.max(image_output_soft_2.detach(), dim=1)[0]
            # pseduo_mask_2 = (normalize(image_output_2) > (1-p**2)).float()
            # pseduo_masknew_2 = pseduo_mask_2 * image_output_2
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            image_out_top2_soft_2 = pseudo_variance_gate(image_output_2)
            image_out_2_entropy = entropy_from_softmax(image_output_soft_2)

            # if  000 <= iter_num <= 80000:
            #     _, image_mask_en_ul_1 = generate_masks(image_output_max_1, image_out_1_entropy,B3, k=0.2)
            #     _, image_mask_en_ul_2 = generate_masks(image_output_max_2, image_out_2_entropy,B3, k=0.2)
            #     image_mask_en = (image_mask_en_ul_1 * image_mask_en_ul_2).bool()
            #     image_patch_last_en = image_patch_last * image_mask_en
            #     image_out_en_1,_ = model_1(image_patch_last_en.unsqueeze(1))
            #     image_out_en_1_soft = torch.softmax(image_out_en_1, dim=1)
            #     image_out_en_pseudo_1 = torch.argmax(image_out_en_1_soft.detach(), dim=1, keepdim=False)

            #     image_out_en_2,_ = model_2(image_patch_last_en.unsqueeze(1))
            #     image_out_en_2_soft = torch.softmax(image_out_en_2, dim=1)
            #     image_out_en_pseudo_2 = torch.argmax(image_out_en_2_soft.detach(), dim=1, keepdim=False)

            #     pseudo_supervision1_en_last = dice_loss(image_out_en_1_soft, image_out_en_pseudo_2.unsqueeze(1),mask = image_mask_en.unsqueeze(1))  
            #     pseudo_supervision2_en_last = dice_loss(image_out_en_2_soft, image_out_en_pseudo_1.unsqueeze(1), mask = image_mask_en.unsqueeze(1)) 
            # else:
            #     pseudo_supervision1_en_last = 0
            #     pseudo_supervision2_en_last = 0
            # Model1 & Model2 Second Step Cross Pseudo Supervision

            start_x = 128 -  p_small[p2]
            start_y = 128 -  p_small[p2]
            end_x = 128 +  p_small[p2]
            end_y = 128 +  p_small[p2]
            
            device = image_output_soft_1.device
            
            r = int(p_small[p2])
            y = torch.arange(-r, r, dtype=torch.float32).view(-1, 1)  # 列向量 [2r, 1]
            x = torch.arange(-r, r, dtype=torch.float32).view(1, -1)  # 行向量 [1, 2r]
            circle_mask = (x.pow(2) + y.pow(2)) <= r**2
            mask_4d = circle_mask.unsqueeze(0).unsqueeze(0)  # 形狀 [1,1,2r,2r]
            mask_3d = circle_mask.unsqueeze(0)               # 形狀 [1,2r,2r]
            mask_4d = mask_4d.to(device)
            mask_3d = mask_3d.to(device)
            region_1 = image_output_soft_1[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            region_2 = image_output_soft_2[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            pseudo_region_1 = pseudo_image_output_1[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()
            pseudo_region_2 = pseudo_image_output_2[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()

            pseudo_supervision6 = dice_loss(region_1, pseudo_region_2.unsqueeze(1))
            pseudo_supervision5 = dice_loss(region_2, pseudo_region_1.unsqueeze(1))
            
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3 + pseudo_supervision6
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4 + pseudo_supervision5
            # loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3  + pseudo_supervision1_en_last
            # loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4  + pseudo_supervision2_en_last
            
            loss = loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer_ema.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()
            # optimizer_ema.step()

            iter_num += 1
            if iter_num % 2 ==0:
                update_ema_variables(model_2, ema_model, 0.99, iter_num)
            else:
                update_ema_variables(model_1, ema_model, 0.99, iter_num)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/model1_loss', loss_1, iter_num)
            writer.add_scalar('info/model2_loss', loss_2, iter_num)
            writer.add_scalar('info/model1/mix_dice', loss_dice_1, iter_num)
            writer.add_scalar('info/model1/mix_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/model2/mix_dice', loss_dice_2, iter_num)
            writer.add_scalar('info/model2/mix_ce', loss_ce_2, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f, min_entropy: %f' % (iter_num, loss, loss_1, loss_2, min_entropy))

            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)

                if performance1 > best_performance1:
                    model_1_stop = 0
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_mode_path)
                    torch.save(model_1.state_dict(), save_best_path)
                    # save_net_opt(model_1, optimizer1, save_mode_path)
                    # save_net_opt(model_1,optimizer1, save_best_path)
                else:
                    model_1_stop += 1
                logging.info('iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_2,classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)

                if performance2 > best_performance2:
                    model_2_stop = 0
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_mode_path)
                    torch.save(model_2.state_dict(), save_best_path)
                    # save_net_opt(model_2, optimizer2, save_mode_path)
                    # save_net_opt(model_2,optimizer2, save_best_path)
                else:
                    model_2_stop += 1
                logging.info('iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model_2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_second_train(args ,pre_snapshot_path, snapshot_path):

    def get_comp_loss(weak, strong):
        """get complementary loss and adaptive sample weight.
        Compares least likely prediction (from strong augment) with argmin of weak augment.

        Args:
            weak (batch): weakly augmented batch
            strong (batch): strongly augmented batch

        Returns:
            comp_loss, as_weight
        """
        ce_loss = CrossEntropyLoss()
        il_output = torch.reshape(
            strong,
            (
                args.batch_size,
                args.num_classes,
                args.image_size[0] * args.image_size[1],
            ),
        )
        # calculate entropy for image-level preds (tensor of length labeled_bs)
        as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.image_size[0] * args.image_size[1]))
        # batch level average of entropy
        as_weight = torch.mean(as_weight)
        # complementary loss
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        comp_loss = as_weight * ce_loss(
            torch.add(torch.negative(strong), 1),
            comp_labels,
        )
        return comp_loss, as_weight

    def normalize(tensor):
            min_val = tensor.min(1, keepdim=True)[0]
            max_val = tensor.max(1, keepdim=True)[0]
            result = tensor - min_val
            result = result / max_val
            return result
        
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model2.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model_1 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[0.5, 0.5, 0.0, 0.0])
    model_2 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[0.5, 0.5, 0.0, 0.0])
    # model_2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()
    ema_model = BCP_net_changedropout(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Train labeled {} samples".format(labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices//2))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer_ema = optim.SGD(ema_model.decoder_sdf.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    model_1.load_state_dict(torch.load(pre_trained_model))
    model_2.load_state_dict(torch.load(pre_trained_model))
    ema_model.load_state_dict(torch.load(pre_trained_model))
    # load_net(ema_model, pre_trained_model)
    # load_net_opt(model_1, optimizer1, pre_trained_model)
    # load_net_opt(model_2, optimizer2, pre_trained_model)
    # model_2.load_from(config)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_1.train()
    model_2.train()
    ema_model.train()

    iter_num = 0
    model_1_stop = 0
    model_2_stop = 0
    p_small = [64, 32, 16]
    p2 = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    min_entropy = 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # print(volume_batch.shape)
            # print(label_batch.shape)
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]


            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a,_ = ema_model(uimg_a)
                pre_b,_ = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b = get_ACDC_masks(pre_b, nms=1)


                pre_a_s,_ = ema_model(uimg_a_s)
                pre_b_s,_ = ema_model(uimg_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)

                img_mask, loss_mask = generate_mask(img_a)
                # unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                # l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)


            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0) 

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask) 
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            # Model1 Loss
            out_unl_1, out_unl_1_result = model_1(net_input_unl_1)
            out_l_1 , out_l_1_result= model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            B1, _, _, _ = out_1.size()
            B2, _, _, _ = net_input_unl_1.size()
            out_soft_1 = torch.softmax(out_1, dim=1)
            C = out_soft_1.size(1)  # 获取类别数
            normalization = torch.log(torch.tensor(C, device=out_soft_1.device))  # 理论最大熵
            mean_entropy_1 = -torch.mean(
                torch.sum(out_soft_1 * torch.log(out_soft_1 + 1e-10), dim=1)
            ) / normalization
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            out_top2_soft_1 = pseudo_variance_gate(out_1)
            out_1_entropy = entropy_from_softmax(out_soft_1)
            out_1_result = torch.cat([out_unl_1_result, out_l_1_result], dim=0)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_label_1 = torch.max(out_soft_1.detach(), dim=1)[1]
            
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2, out_unl_2_result = model_2(net_input_unl_2)
            out_l_2, out_l_2_result = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_top2_soft_2 = pseudo_variance_gate(out_2)
            out_soft_2 = torch.softmax(out_2, dim=1)
            C = out_soft_2.size(1)  
            normalization = torch.log(torch.tensor(C, device=out_soft_2.device))
            mean_entropy_2 = -torch.mean(
                torch.sum(out_soft_2 * torch.log(out_soft_2 + 1e-10), dim=1)
            ) / normalization
            out_2_result = torch.cat([out_unl_2_result, out_l_2_result], dim=0)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_label_2 = torch.max(out_soft_2.detach(), dim=1)[1]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False) 
            out_2_entropy = entropy_from_softmax(out_soft_2)
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            mean_entropy = (mean_entropy_1 + mean_entropy_2) / 2
            if mean_entropy < min_entropy :
                min_entropy = mean_entropy

            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))  
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1)) 
            #     p2 = (p2 + 1) % 3
            p = iter_num/max_iterations

            image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1_result, out_2_result,args)
            # image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)

            # print(out_max_1.shape)
            # image_patch_last = Ncut(out_max_1, out_max_2, net_input_1, net_input_2, p, args)
            B3, _, _ = image_patch_last.size()
            image_output_1,_ = model_1(image_patch_last.unsqueeze(1))  
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            # pseduo_mask_1 = (normalize(image_output_1) > (1-p**2)).float()
            # pseduo_masknew_1 = pseduo_mask_1 * image_output_1
            image_output_max_1 = torch.max(image_output_soft_1.detach(), dim=1)[0]
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            # image_out_top2_soft_1 = pseudo_variance_gate(image_output_1)
            # image_out_1_entropy = entropy_from_softmax(image_output_soft_1)
            image_output_2,_ = model_2(image_patch_last.unsqueeze(1))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            image_output_max_2 = torch.max(image_output_soft_2.detach(), dim=1)[0]
            # pseduo_mask_2 = (normalize(image_output_2) > (1-p**2)).float()
            # pseduo_masknew_2 = pseduo_mask_2 * image_output_2
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            # image_out_top2_soft_2 = pseudo_variance_gate(image_output_2)
            # image_out_2_entropy = entropy_from_softmax(image_output_soft_2)

            # if  000 <= iter_num <= 80000:
            #     _, image_mask_en_ul_1 = generate_masks(image_output_max_1, image_out_1_entropy,B3, k=0.2)
            #     _, image_mask_en_ul_2 = generate_masks(image_output_max_2, image_out_2_entropy,B3, k=0.2)
            #     image_mask_en = (image_mask_en_ul_1 * image_mask_en_ul_2).bool()
            #     image_patch_last_en = image_patch_last * image_mask_en
            #     image_out_en_1,_ = model_1(image_patch_last_en.unsqueeze(1))
            #     image_out_en_1_soft = torch.softmax(image_out_en_1, dim=1)
            #     image_out_en_pseudo_1 = torch.argmax(image_out_en_1_soft.detach(), dim=1, keepdim=False)

            #     image_out_en_2,_ = model_2(image_patch_last_en.unsqueeze(1))
            #     image_out_en_2_soft = torch.softmax(image_out_en_2, dim=1)
            #     image_out_en_pseudo_2 = torch.argmax(image_out_en_2_soft.detach(), dim=1, keepdim=False)

            #     pseudo_supervision1_en_last = dice_loss(image_out_en_1_soft, image_out_en_pseudo_2.unsqueeze(1),mask = image_mask_en.unsqueeze(1))  
            #     pseudo_supervision2_en_last = dice_loss(image_out_en_2_soft, image_out_en_pseudo_1.unsqueeze(1), mask = image_mask_en.unsqueeze(1)) 
            # else:
            #     pseudo_supervision1_en_last = 0
            #     pseudo_supervision2_en_last = 0
            # Model1 & Model2 Second Step Cross Pseudo Supervision

            start_x = 128 -  p_small[p2]
            start_y = 128 -  p_small[p2]
            end_x = 128 +  p_small[p2]
            end_y = 128 +  p_small[p2]
            
            device = image_output_soft_1.device
            
            r = int(p_small[p2])
            y = torch.arange(-r, r, dtype=torch.float32).view(-1, 1)  # 列向量 [2r, 1]
            x = torch.arange(-r, r, dtype=torch.float32).view(1, -1)  # 行向量 [1, 2r]
            circle_mask = (x.pow(2) + y.pow(2)) <= r**2
            mask_4d = circle_mask.unsqueeze(0).unsqueeze(0)  # 形狀 [1,1,2r,2r]
            mask_3d = circle_mask.unsqueeze(0)               # 形狀 [1,2r,2r]
            mask_4d = mask_4d.to(device)
            mask_3d = mask_3d.to(device)
            region_1 = image_output_soft_1[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            region_2 = image_output_soft_2[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            pseudo_region_1 = pseudo_image_output_1[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()
            pseudo_region_2 = pseudo_image_output_2[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()

            pseudo_supervision6 = dice_loss(region_1, pseudo_region_2.unsqueeze(1))
            pseudo_supervision5 = dice_loss(region_2, pseudo_region_1.unsqueeze(1))
            
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3 + pseudo_supervision6
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4 + pseudo_supervision5
            # loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3  + pseudo_supervision1_en_last
            # loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4  + pseudo_supervision2_en_last
            
            loss = loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer_ema.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()
            # optimizer_ema.step()

            iter_num += 1
            if iter_num % 2 ==0:
                update_ema_variables(model_2, ema_model, 0.99, iter_num)
            else:
                update_ema_variables(model_1, ema_model, 0.99, iter_num)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/model1_loss', loss_1, iter_num)
            writer.add_scalar('info/model2_loss', loss_2, iter_num)
            writer.add_scalar('info/model1/mix_dice', loss_dice_1, iter_num)
            writer.add_scalar('info/model1/mix_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/model2/mix_dice', loss_dice_2, iter_num)
            writer.add_scalar('info/model2/mix_ce', loss_ce_2, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            # logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f' % (iter_num, loss, loss_1, loss_2))
            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f, min_entropy: %f' % (iter_num, loss, loss_1, loss_2, min_entropy))

            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)

                if performance1 > best_performance1:
                    model_1_stop = 0
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_mode_path)
                    torch.save(model_1.state_dict(), save_best_path)
                    # save_net_opt(model_1, optimizer1, save_mode_path)
                    # save_net_opt(model_1,optimizer1, save_best_path)
                else:
                    model_1_stop += 1
                logging.info('iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_2,classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)

                if performance2 > best_performance2:
                    model_2_stop = 0
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_mode_path)
                    torch.save(model_2.state_dict(), save_best_path)
                    # save_net_opt(model_2, optimizer2, save_mode_path)
                    # save_net_opt(model_2,optimizer2, save_best_path)
                else:
                    model_2_stop += 1
                logging.info('iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model_2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_third_train(args ,pre_snapshot_path, snapshot_path):

    def get_comp_loss(weak, strong):
        """get complementary loss and adaptive sample weight.
        Compares least likely prediction (from strong augment) with argmin of weak augment.

        Args:
            weak (batch): weakly augmented batch
            strong (batch): strongly augmented batch

        Returns:
            comp_loss, as_weight
        """
        ce_loss = CrossEntropyLoss()
        il_output = torch.reshape(
            strong,
            (
                args.batch_size,
                args.num_classes,
                args.image_size[0] * args.image_size[1],
            ),
        )
        # calculate entropy for image-level preds (tensor of length labeled_bs)
        as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.image_size[0] * args.image_size[1]))
        # batch level average of entropy
        as_weight = torch.mean(as_weight)
        # complementary loss
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        comp_loss = as_weight * ce_loss(
            torch.add(torch.negative(strong), 1),
            comp_labels,
        )
        return comp_loss, as_weight

    def normalize(tensor):
            min_val = tensor.min(1, keepdim=True)[0]
            max_val = tensor.max(1, keepdim=True)[0]
            result = tensor - min_val
            result = result / max_val
            return result
        
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model1.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model_1 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[0.0, 0.5, 0.5, 0.0])
    model_2 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[0.0, 0.5, 0.5, 0.0])
    # model_2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()
    ema_model = BCP_net_changedropout(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Train labeled {} samples".format(labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices//2))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer_ema = optim.SGD(ema_model.decoder_sdf.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    model_1.load_state_dict(torch.load(pre_trained_model))
    model_2.load_state_dict(torch.load(pre_trained_model))
    ema_model.load_state_dict(torch.load(pre_trained_model))
    # load_net(ema_model, pre_trained_model)
    # load_net_opt(model_1, optimizer1, pre_trained_model)
    # load_net_opt(model_2, optimizer2, pre_trained_model)
    # model_2.load_from(config)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_1.train()
    model_2.train()
    ema_model.train()

    iter_num = 0
    model_1_stop = 0
    model_2_stop = 0
    p_small = [64, 32, 16]
    p2 = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    min_entropy = 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # print(volume_batch.shape)
            # print(label_batch.shape)
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]


            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a,_ = ema_model(uimg_a)
                pre_b,_ = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b = get_ACDC_masks(pre_b, nms=1)


                pre_a_s,_ = ema_model(uimg_a_s)
                pre_b_s,_ = ema_model(uimg_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)


                img_mask, loss_mask = generate_mask(img_a)
                # unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                # l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)


            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0) 

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask) 
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            # Model1 Loss
            out_unl_1, out_unl_1_result = model_1(net_input_unl_1)
            out_l_1 , out_l_1_result= model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            B1, _, _, _ = out_1.size()
            B2, _, _, _ = net_input_unl_1.size()
            out_soft_1 = torch.softmax(out_1, dim=1)
            C = out_soft_1.size(1)  # 获取类别数
            normalization = torch.log(torch.tensor(C, device=out_soft_1.device))  # 理论最大熵
            mean_entropy_1 = -torch.mean(
                torch.sum(out_soft_1 * torch.log(out_soft_1 + 1e-10), dim=1)
            ) / normalization
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            out_top2_soft_1 = pseudo_variance_gate(out_1)
            out_1_entropy = entropy_from_softmax(out_soft_1)
            out_1_result = torch.cat([out_unl_1_result, out_l_1_result], dim=0)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_label_1 = torch.max(out_soft_1.detach(), dim=1)[1]
            
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2, out_unl_2_result = model_2(net_input_unl_2)
            out_l_2, out_l_2_result = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_top2_soft_2 = pseudo_variance_gate(out_2)
            out_soft_2 = torch.softmax(out_2, dim=1)
            C = out_soft_2.size(1)  
            normalization = torch.log(torch.tensor(C, device=out_soft_2.device))
            mean_entropy_2 = -torch.mean(
                torch.sum(out_soft_2 * torch.log(out_soft_2 + 1e-10), dim=1)
            ) / normalization
            out_2_result = torch.cat([out_unl_2_result, out_l_2_result], dim=0)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_label_2 = torch.max(out_soft_2.detach(), dim=1)[1]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False) 
            out_2_entropy = entropy_from_softmax(out_soft_2)
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            mean_entropy = (mean_entropy_1 + mean_entropy_2) / 2
            if mean_entropy < min_entropy :
                min_entropy = mean_entropy

            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))  
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1)) 
            # pseudo_supervision1_top2 = dice_loss(out_soft_1, out_2_label_init.unsqueeze(1).float(), mask=mask_3.unsqueeze(1))  
            # pseudo_supervision2_top2 = dice_loss(out_soft_2, out_1_label_init.unsqueeze(1).float(), mask=mask_3_2.unsqueeze(1))  
            # if model_1_stop == 100 or model_2_stop == 100:
            #     p2 = (p2 + 1) % 3
            p = iter_num/max_iterations

            image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1_result, out_2_result,args)
            # image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)

            # print(out_max_1.shape)
            # image_patch_last = Ncut(out_max_1, out_max_2, net_input_1, net_input_2, p, args)
            B3, _, _ = image_patch_last.size()
            image_output_1,_ = model_1(image_patch_last.unsqueeze(1))  
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            # pseduo_mask_1 = (normalize(image_output_1) > (1-p**2)).float()
            # pseduo_masknew_1 = pseduo_mask_1 * image_output_1
            image_output_max_1 = torch.max(image_output_soft_1.detach(), dim=1)[0]
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_out_top2_soft_1 = pseudo_variance_gate(image_output_1)
            image_out_1_entropy = entropy_from_softmax(image_output_soft_1)
            image_output_2,_ = model_2(image_patch_last.unsqueeze(1))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            image_output_max_2 = torch.max(image_output_soft_2.detach(), dim=1)[0]
            # pseduo_mask_2 = (normalize(image_output_2) > (1-p**2)).float()
            # pseduo_masknew_2 = pseduo_mask_2 * image_output_2
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            image_out_top2_soft_2 = pseudo_variance_gate(image_output_2)
            image_out_2_entropy = entropy_from_softmax(image_output_soft_2)

            # if  000 <= iter_num <= 80000:
            #     _, image_mask_en_ul_1 = generate_masks(image_output_max_1, image_out_1_entropy,B3, k=0.2)
            #     _, image_mask_en_ul_2 = generate_masks(image_output_max_2, image_out_2_entropy,B3, k=0.2)
            #     image_mask_en = (image_mask_en_ul_1 * image_mask_en_ul_2).bool()
            #     image_patch_last_en = image_patch_last * image_mask_en
            #     image_out_en_1,_ = model_1(image_patch_last_en.unsqueeze(1))
            #     image_out_en_1_soft = torch.softmax(image_out_en_1, dim=1)
            #     image_out_en_pseudo_1 = torch.argmax(image_out_en_1_soft.detach(), dim=1, keepdim=False)

            #     image_out_en_2,_ = model_2(image_patch_last_en.unsqueeze(1))
            #     image_out_en_2_soft = torch.softmax(image_out_en_2, dim=1)
            #     image_out_en_pseudo_2 = torch.argmax(image_out_en_2_soft.detach(), dim=1, keepdim=False)

            #     pseudo_supervision1_en_last = dice_loss(image_out_en_1_soft, image_out_en_pseudo_2.unsqueeze(1),mask = image_mask_en.unsqueeze(1))  
            #     pseudo_supervision2_en_last = dice_loss(image_out_en_2_soft, image_out_en_pseudo_1.unsqueeze(1), mask = image_mask_en.unsqueeze(1)) 
            # else:
            #     pseudo_supervision1_en_last = 0
            #     pseudo_supervision2_en_last = 0
            # Model1 & Model2 Second Step Cross Pseudo Supervision

            start_x = 128 -  p_small[p2]
            start_y = 128 -  p_small[p2]
            end_x = 128 +  p_small[p2]
            end_y = 128 +  p_small[p2]
            
            device = image_output_soft_1.device
            
            r = int(p_small[p2])
            y = torch.arange(-r, r, dtype=torch.float32).view(-1, 1)  # 列向量 [2r, 1]
            x = torch.arange(-r, r, dtype=torch.float32).view(1, -1)  # 行向量 [1, 2r]
            circle_mask = (x.pow(2) + y.pow(2)) <= r**2
            mask_4d = circle_mask.unsqueeze(0).unsqueeze(0)  # 形狀 [1,1,2r,2r]
            mask_3d = circle_mask.unsqueeze(0)               # 形狀 [1,2r,2r]
            mask_4d = mask_4d.to(device)
            mask_3d = mask_3d.to(device)
            region_1 = image_output_soft_1[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            region_2 = image_output_soft_2[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            pseudo_region_1 = pseudo_image_output_1[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()
            pseudo_region_2 = pseudo_image_output_2[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()

            pseudo_supervision6 = dice_loss(region_1, pseudo_region_2.unsqueeze(1))
            pseudo_supervision5 = dice_loss(region_2, pseudo_region_1.unsqueeze(1))
            
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3 + pseudo_supervision6
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4 + pseudo_supervision5
            # loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3  + pseudo_supervision1_en_last
            # loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4  + pseudo_supervision2_en_last
            
            loss = loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer_ema.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()
            # optimizer_ema.step()

            iter_num += 1
            if iter_num % 2 ==0:
                update_ema_variables(model_2, ema_model, 0.99, iter_num)
            else:
                update_ema_variables(model_1, ema_model, 0.99, iter_num)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/model1_loss', loss_1, iter_num)
            writer.add_scalar('info/model2_loss', loss_2, iter_num)
            writer.add_scalar('info/model1/mix_dice', loss_dice_1, iter_num)
            writer.add_scalar('info/model1/mix_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/model2/mix_dice', loss_dice_2, iter_num)
            writer.add_scalar('info/model2/mix_ce', loss_ce_2, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            # logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f' % (iter_num, loss, loss_1, loss_2))
            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f, min_entropy: %f' % (iter_num, loss, loss_1, loss_2, min_entropy))

            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)

                if performance1 > best_performance1:
                    model_1_stop = 0
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_mode_path)
                    torch.save(model_1.state_dict(), save_best_path)
                    # save_net_opt(model_1, optimizer1, save_mode_path)
                    # save_net_opt(model_1,optimizer1, save_best_path)
                else:
                    model_1_stop += 1
                logging.info('iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_2,classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)

                if performance2 > best_performance2:
                    model_2_stop = 0
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_mode_path)
                    torch.save(model_2.state_dict(), save_best_path)
                    # save_net_opt(model_2, optimizer2, save_mode_path)
                    # save_net_opt(model_2,optimizer2, save_best_path)
                else:
                    model_2_stop += 1
                logging.info('iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model_2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_fourth_train(args ,pre_snapshot_path, snapshot_path):

    def get_comp_loss(weak, strong):
        """get complementary loss and adaptive sample weight.
        Compares least likely prediction (from strong augment) with argmin of weak augment.

        Args:
            weak (batch): weakly augmented batch
            strong (batch): strongly augmented batch

        Returns:
            comp_loss, as_weight
        """
        ce_loss = CrossEntropyLoss()
        il_output = torch.reshape(
            strong,
            (
                args.batch_size,
                args.num_classes,
                args.image_size[0] * args.image_size[1],
            ),
        )
        # calculate entropy for image-level preds (tensor of length labeled_bs)
        as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.image_size[0] * args.image_size[1]))
        # batch level average of entropy
        as_weight = torch.mean(as_weight)
        # complementary loss
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        comp_loss = as_weight * ce_loss(
            torch.add(torch.negative(strong), 1),
            comp_labels,
        )
        return comp_loss, as_weight

    def normalize(tensor):
            min_val = tensor.min(1, keepdim=True)[0]
            max_val = tensor.max(1, keepdim=True)[0]
            result = tensor - min_val
            result = result / max_val
            return result
        
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model1.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model_1 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[0.0, 0.0, 0.5, 0.5])
    model_2 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[0.0, 0.0, 0.5, 0.5])
    # model_2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()
    ema_model = BCP_net_changedropout(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Train labeled {} samples".format(labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices//2))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer_ema = optim.SGD(ema_model.decoder_sdf.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    model_1.load_state_dict(torch.load(pre_trained_model))
    model_2.load_state_dict(torch.load(pre_trained_model))
    ema_model.load_state_dict(torch.load(pre_trained_model))
    # load_net(ema_model, pre_trained_model)
    # load_net_opt(model_1, optimizer1, pre_trained_model)
    # load_net_opt(model_2, optimizer2, pre_trained_model)
    # model_2.load_from(config)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_1.train()
    model_2.train()
    ema_model.train()

    iter_num = 0
    model_1_stop = 0
    model_2_stop = 0
    p_small = [64, 32, 16]
    p2 = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    min_entropy = 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # print(volume_batch.shape)
            # print(label_batch.shape)
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]


            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a,_ = ema_model(uimg_a)
                pre_b,_ = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b = get_ACDC_masks(pre_b, nms=1)

                # pre_a = model_2(uimg_a)
                # pre_b = model_2(uimg_b)
                # plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                # plab_b = get_ACDC_masks(pre_b, nms=1)

                # plab_a_sdf = compute_sdf_tensor(plab_a)
                # plab_a_sdf = normalize_sdf(plab_a_sdf)
                # plab_a_sdf = plab_a_sdf.cuda()

                # plab_b_sdf = compute_sdf_tensor(plab_b)
                # plab_b_sdf = normalize_sdf(plab_b_sdf)
                # plab_b_sdf = plab_b_sdf.cuda()

                pre_a_s,_ = ema_model(uimg_a_s)
                pre_b_s,_ = ema_model(uimg_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)

                # pre_a_s = model_2(uimg_a_s)
                # pre_b_s = model_2(uimg_b_s)
                # plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                # plab_b_s = get_ACDC_masks(pre_b_s, nms=1)

                # plab_a_s_sdf = compute_sdf_tensor(plab_a_s)
                # plab_a_s_sdf = normalize_sdf(plab_a_s_sdf)
                # plab_a_s_sdf = plab_a_s_sdf.cuda()

                # plab_b_s_sdf = compute_sdf_tensor(plab_b_s)
                # plab_b_s_sdf = normalize_sdf(plab_b_s_sdf)
                # plab_b_s_sdf = plab_b_s_sdf.cuda()

                img_mask, loss_mask = generate_mask(img_a)
                # unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                # l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)


            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0) 

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask) 
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            # Model1 Loss
            out_unl_1, out_unl_1_result = model_1(net_input_unl_1)
            out_l_1 , out_l_1_result= model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            B1, _, _, _ = out_1.size()
            B2, _, _, _ = net_input_unl_1.size()
            out_soft_1 = torch.softmax(out_1, dim=1)
            C = out_soft_1.size(1)  # 获取类别数
            normalization = torch.log(torch.tensor(C, device=out_soft_1.device))  # 理论最大熵
            mean_entropy_1 = -torch.mean(
                torch.sum(out_soft_1 * torch.log(out_soft_1 + 1e-10), dim=1)
            ) / normalization
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            out_top2_soft_1 = pseudo_variance_gate(out_1)
            out_1_entropy = entropy_from_softmax(out_soft_1)
            # mask_1 = torch.where((out_1_entropy >= 1) & (out_1_entropy < 2), torch.tensor(1, device=out_1_entropy.device), torch.tensor(0, device=out_1_entropy.device))
            # mask_2 = torch.where((out_1_entropy < 1), torch.tensor(1, device=out_1_entropy.device), torch.tensor(0, device=out_1_entropy.device))
            # out_1_label_init = out_pseudo_1 * mask_2 + out_top2_soft_1 * mask_1
            # mask_3 = torch.where((out_1_entropy >= 2), torch.tensor(1, device=out_1_entropy.device), torch.tensor(0, device=out_1_entropy.device))
            out_1_result = torch.cat([out_unl_1_result, out_l_1_result], dim=0)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_label_1 = torch.max(out_soft_1.detach(), dim=1)[1]
            
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2, out_unl_2_result = model_2(net_input_unl_2)
            out_l_2, out_l_2_result = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_top2_soft_2 = pseudo_variance_gate(out_2)
            out_soft_2 = torch.softmax(out_2, dim=1)
            C = out_soft_2.size(1)  
            normalization = torch.log(torch.tensor(C, device=out_soft_2.device))
            mean_entropy_2 = -torch.mean(
                torch.sum(out_soft_2 * torch.log(out_soft_2 + 1e-10), dim=1)
            ) / normalization
            out_2_result = torch.cat([out_unl_2_result, out_l_2_result], dim=0)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_label_2 = torch.max(out_soft_2.detach(), dim=1)[1]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False) 
            out_2_entropy = entropy_from_softmax(out_soft_2)
            # mask_1_2 = torch.where((out_2_entropy >= 1) & (out_2_entropy < 2), torch.tensor(1, device=out_2_entropy.device), torch.tensor(0, device=out_2_entropy.device))
            # mask_2_2 = torch.where((out_2_entropy < 1), torch.tensor(1, device=out_2_entropy.device), torch.tensor(0, device=out_2_entropy.device))
            # out_2_label_init = out_pseudo_2 * mask_2_2 + out_top2_soft_2 * mask_1_2
            # mask_3_2 = torch.where((out_2_entropy >= 2), torch.tensor(1, device=out_2_entropy.device), torch.tensor(0, device=out_2_entropy.device))
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            mean_entropy = (mean_entropy_1 + mean_entropy_2) / 2
            if mean_entropy < min_entropy :
                min_entropy = mean_entropy

            # Model1 & Model2 Cross Pseudo Supervision
            # if iter_num >= 15000:
            #     _, mask_en_ul_1 = generate_masks(out_max_1, out_1_entropy,B1)
            #     _, mask_en_ul_2 = generate_masks(out_max_2, out_2_entropy,B1)
            #     mask_en = (mask_en_ul_1 * mask_en_ul_2).bool()
            #     net_input_en_1 = net_input_1 * mask_en.unsqueeze(1)
            #     net_input_en_2 = net_input_2 * mask_en.unsqueeze(1)
            #     net_input_en = torch.cat([net_input_en_1[:B2], net_input_en_2[:B2]], dim=0)
            #     out_en_1,_ = model_1(net_input_en)
            #     out_en_1_soft = torch.softmax(out_en_1, dim=1)
            #     out_en_pseudo_1 = torch.argmax(out_en_1_soft.detach(), dim=1, keepdim=False)

            #     out_en_2,_ = model_2(net_input_en)
            #     out_en_2_soft = torch.softmax(out_en_2, dim=1)
            #     out_en_pseudo_2 = torch.argmax(out_en_2_soft.detach(), dim=1, keepdim=False)

            #     pseudo_supervision1_en = dice_loss(out_en_1_soft, out_en_pseudo_2.unsqueeze(1),mask = mask_en.unsqueeze(1))  
            #     pseudo_supervision2_en = dice_loss(out_en_2_soft, out_en_pseudo_1.unsqueeze(1), mask = mask_en.unsqueeze(1)) 
            # else:
                # pseudo_supervision1_en = 0
                # pseudo_supervision2_en = 0

            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))  
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1)) 
            # pseudo_supervision1_top2 = dice_loss(out_soft_1, out_2_label_init.unsqueeze(1).float(), mask=mask_3.unsqueeze(1))  
            # pseudo_supervision2_top2 = dice_loss(out_soft_2, out_1_label_init.unsqueeze(1).float(), mask=mask_3_2.unsqueeze(1))  
            # if model_1_stop == 100 or model_2_stop == 100:
            #     p2 = (p2 + 1) % 3
            p = iter_num/max_iterations
            # if p <  1/5 :
            #     p2 = 0
            # elif 1/5 <= p < 2/5:
            #     p2 = 1
            # # elif 0.40 <= p < 0.60:
            # #     p2 = 2
            # # elif 0.60 <= p < 0.80:
            # #     p2 = 3
            # elif 2/5 <= p :
            #     p2 = 2
            # ABD-R New Training Sample
            # image_patch_last = ABD_I_Conf(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)
            # image_patch_last = Overlap(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, p, args)
            # image_patch_last = layer(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args, p_small)
            # image_patch_last = layer(out_max_1, out_max_2, out_label_1, out_label_2, net_input_1, net_input_2, out_1, out_2, p, p2, args, p_small)
            # if p <  3/5 :
            #     image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1_result, out_2_result,args)
            # else:
            #     image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)
            image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1_result, out_2_result,args)
            # image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)

            # print(out_max_1.shape)
            # image_patch_last = Ncut(out_max_1, out_max_2, net_input_1, net_input_2, p, args)
            B3, _, _ = image_patch_last.size()
            image_output_1,_ = model_1(image_patch_last.unsqueeze(1))  
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            # pseduo_mask_1 = (normalize(image_output_1) > (1-p**2)).float()
            # pseduo_masknew_1 = pseduo_mask_1 * image_output_1
            image_output_max_1 = torch.max(image_output_soft_1.detach(), dim=1)[0]
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_out_top2_soft_1 = pseudo_variance_gate(image_output_1)
            image_out_1_entropy = entropy_from_softmax(image_output_soft_1)
            image_output_2,_ = model_2(image_patch_last.unsqueeze(1))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            image_output_max_2 = torch.max(image_output_soft_2.detach(), dim=1)[0]
            # pseduo_mask_2 = (normalize(image_output_2) > (1-p**2)).float()
            # pseduo_masknew_2 = pseduo_mask_2 * image_output_2
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            image_out_top2_soft_2 = pseudo_variance_gate(image_output_2)
            image_out_2_entropy = entropy_from_softmax(image_output_soft_2)

            # if  000 <= iter_num <= 80000:
            #     _, image_mask_en_ul_1 = generate_masks(image_output_max_1, image_out_1_entropy,B3, k=0.2)
            #     _, image_mask_en_ul_2 = generate_masks(image_output_max_2, image_out_2_entropy,B3, k=0.2)
            #     image_mask_en = (image_mask_en_ul_1 * image_mask_en_ul_2).bool()
            #     image_patch_last_en = image_patch_last * image_mask_en
            #     image_out_en_1,_ = model_1(image_patch_last_en.unsqueeze(1))
            #     image_out_en_1_soft = torch.softmax(image_out_en_1, dim=1)
            #     image_out_en_pseudo_1 = torch.argmax(image_out_en_1_soft.detach(), dim=1, keepdim=False)

            #     image_out_en_2,_ = model_2(image_patch_last_en.unsqueeze(1))
            #     image_out_en_2_soft = torch.softmax(image_out_en_2, dim=1)
            #     image_out_en_pseudo_2 = torch.argmax(image_out_en_2_soft.detach(), dim=1, keepdim=False)

            #     pseudo_supervision1_en_last = dice_loss(image_out_en_1_soft, image_out_en_pseudo_2.unsqueeze(1),mask = image_mask_en.unsqueeze(1))  
            #     pseudo_supervision2_en_last = dice_loss(image_out_en_2_soft, image_out_en_pseudo_1.unsqueeze(1), mask = image_mask_en.unsqueeze(1)) 
            # else:
            #     pseudo_supervision1_en_last = 0
            #     pseudo_supervision2_en_last = 0
            # Model1 & Model2 Second Step Cross Pseudo Supervision

            # start_x = 128 -  p_small[p2]
            # start_y = 128 -  p_small[p2]
            # end_x = 128 +  p_small[p2]
            # end_y = 128 +  p_small[p2]
            
            # device = image_output_soft_1.device
            
            # r = int(p_small[p2])
            # y = torch.arange(-r, r, dtype=torch.float32).view(-1, 1)  # 列向量 [2r, 1]
            # x = torch.arange(-r, r, dtype=torch.float32).view(1, -1)  # 行向量 [1, 2r]
            # circle_mask = (x.pow(2) + y.pow(2)) <= r**2
            # mask_4d = circle_mask.unsqueeze(0).unsqueeze(0)  # 形狀 [1,1,2r,2r]
            # mask_3d = circle_mask.unsqueeze(0)               # 形狀 [1,2r,2r]
            # mask_4d = mask_4d.to(device)
            # mask_3d = mask_3d.to(device)
            # region_1 = image_output_soft_1[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            # region_2 = image_output_soft_2[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            # pseudo_region_1 = pseudo_image_output_1[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()
            # pseudo_region_2 = pseudo_image_output_2[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()

            # pseudo_supervision6 = dice_loss(region_1, pseudo_region_2.unsqueeze(1))
            # pseudo_supervision5 = dice_loss(region_2, pseudo_region_1.unsqueeze(1))
            
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4
            # loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3  + pseudo_supervision1_en_last
            # loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4  + pseudo_supervision2_en_last
            
            loss = loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer_ema.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()
            # optimizer_ema.step()

            iter_num += 1
            if iter_num % 2 ==0:
                update_ema_variables(model_2, ema_model, 0.99, iter_num)
            else:
                update_ema_variables(model_1, ema_model, 0.99, iter_num)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/model1_loss', loss_1, iter_num)
            writer.add_scalar('info/model2_loss', loss_2, iter_num)
            writer.add_scalar('info/model1/mix_dice', loss_dice_1, iter_num)
            writer.add_scalar('info/model1/mix_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/model2/mix_dice', loss_dice_2, iter_num)
            writer.add_scalar('info/model2/mix_ce', loss_ce_2, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            # logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f' % (iter_num, loss, loss_1, loss_2))
            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f, min_entropy: %f' % (iter_num, loss, loss_1, loss_2, min_entropy))

            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)

                if performance1 > best_performance1:
                    model_1_stop = 0
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_mode_path)
                    torch.save(model_1.state_dict(), save_best_path)
                    # save_net_opt(model_1, optimizer1, save_mode_path)
                    # save_net_opt(model_1,optimizer1, save_best_path)
                else:
                    model_1_stop += 1
                logging.info('iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_2,classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)

                if performance2 > best_performance2:
                    model_2_stop = 0
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_mode_path)
                    torch.save(model_2.state_dict(), save_best_path)
                    # save_net_opt(model_2, optimizer2, save_mode_path)
                    # save_net_opt(model_2,optimizer2, save_best_path)
                else:
                    model_2_stop += 1
                logging.info('iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model_2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_last_train(args ,pre_snapshot_path, snapshot_path):

    def get_comp_loss(weak, strong):
        """get complementary loss and adaptive sample weight.
        Compares least likely prediction (from strong augment) with argmin of weak augment.

        Args:
            weak (batch): weakly augmented batch
            strong (batch): strongly augmented batch

        Returns:
            comp_loss, as_weight
        """
        ce_loss = CrossEntropyLoss()
        il_output = torch.reshape(
            strong,
            (
                args.batch_size,
                args.num_classes,
                args.image_size[0] * args.image_size[1],
            ),
        )
        # calculate entropy for image-level preds (tensor of length labeled_bs)
        as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.image_size[0] * args.image_size[1]))
        # batch level average of entropy
        as_weight = torch.mean(as_weight)
        # complementary loss
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        comp_loss = as_weight * ce_loss(
            torch.add(torch.negative(strong), 1),
            comp_labels,
        )
        return comp_loss, as_weight

    def normalize(tensor):
            min_val = tensor.min(1, keepdim=True)[0]
            max_val = tensor.max(1, keepdim=True)[0]
            result = tensor - min_val
            result = result / max_val
            return result
        
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model1.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model_1 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[0.0, 0.0, 0.0, 1.0])
    model_2 = BCP_net_changedropout(in_chns=1, class_num=num_classes, weights=[0.0, 0.0, 0.0, 1.0])
    # model_2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()
    ema_model = BCP_net_changedropout(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Train labeled {} samples".format(labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices//2))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer_ema = optim.SGD(ema_model.decoder_sdf.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    model_1.load_state_dict(torch.load(pre_trained_model))
    model_2.load_state_dict(torch.load(pre_trained_model))
    ema_model.load_state_dict(torch.load(pre_trained_model))
    # load_net(ema_model, pre_trained_model)
    # load_net_opt(model_1, optimizer1, pre_trained_model)
    # load_net_opt(model_2, optimizer2, pre_trained_model)
    # model_2.load_from(config)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_1.train()
    model_2.train()
    ema_model.train()

    iter_num = 0
    model_1_stop = 0
    model_2_stop = 0
    p_small = [64, 32, 16]
    p2 = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    min_entropy = 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # print(volume_batch.shape)
            # print(label_batch.shape)
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]


            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a,_ = ema_model(uimg_a)
                pre_b,_ = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b = get_ACDC_masks(pre_b, nms=1)

                # pre_a = model_2(uimg_a)
                # pre_b = model_2(uimg_b)
                # plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                # plab_b = get_ACDC_masks(pre_b, nms=1)

                # plab_a_sdf = compute_sdf_tensor(plab_a)
                # plab_a_sdf = normalize_sdf(plab_a_sdf)
                # plab_a_sdf = plab_a_sdf.cuda()

                # plab_b_sdf = compute_sdf_tensor(plab_b)
                # plab_b_sdf = normalize_sdf(plab_b_sdf)
                # plab_b_sdf = plab_b_sdf.cuda()

                pre_a_s,_ = ema_model(uimg_a_s)
                pre_b_s,_ = ema_model(uimg_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)

                # pre_a_s = model_2(uimg_a_s)
                # pre_b_s = model_2(uimg_b_s)
                # plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                # plab_b_s = get_ACDC_masks(pre_b_s, nms=1)

                # plab_a_s_sdf = compute_sdf_tensor(plab_a_s)
                # plab_a_s_sdf = normalize_sdf(plab_a_s_sdf)
                # plab_a_s_sdf = plab_a_s_sdf.cuda()

                # plab_b_s_sdf = compute_sdf_tensor(plab_b_s)
                # plab_b_s_sdf = normalize_sdf(plab_b_s_sdf)
                # plab_b_s_sdf = plab_b_s_sdf.cuda()

                img_mask, loss_mask = generate_mask(img_a)
                # unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                # l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)


            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0) 

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask) 
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            # Model1 Loss
            out_unl_1, out_unl_1_result = model_1(net_input_unl_1)
            out_l_1 , out_l_1_result= model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            B1, _, _, _ = out_1.size()
            B2, _, _, _ = net_input_unl_1.size()
            out_soft_1 = torch.softmax(out_1, dim=1)
            C = out_soft_1.size(1)  # 获取类别数
            normalization = torch.log(torch.tensor(C, device=out_soft_1.device))  # 理论最大熵
            mean_entropy_1 = -torch.mean(
                torch.sum(out_soft_1 * torch.log(out_soft_1 + 1e-10), dim=1)
            ) / normalization
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            out_top2_soft_1 = pseudo_variance_gate(out_1)
            out_1_entropy = entropy_from_softmax(out_soft_1)
            # mask_1 = torch.where((out_1_entropy >= 1) & (out_1_entropy < 2), torch.tensor(1, device=out_1_entropy.device), torch.tensor(0, device=out_1_entropy.device))
            # mask_2 = torch.where((out_1_entropy < 1), torch.tensor(1, device=out_1_entropy.device), torch.tensor(0, device=out_1_entropy.device))
            # out_1_label_init = out_pseudo_1 * mask_2 + out_top2_soft_1 * mask_1
            # mask_3 = torch.where((out_1_entropy >= 2), torch.tensor(1, device=out_1_entropy.device), torch.tensor(0, device=out_1_entropy.device))
            out_1_result = torch.cat([out_unl_1_result, out_l_1_result], dim=0)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_label_1 = torch.max(out_soft_1.detach(), dim=1)[1]
            
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2, out_unl_2_result = model_2(net_input_unl_2)
            out_l_2, out_l_2_result = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_top2_soft_2 = pseudo_variance_gate(out_2)
            out_soft_2 = torch.softmax(out_2, dim=1)
            C = out_soft_2.size(1)  
            normalization = torch.log(torch.tensor(C, device=out_soft_2.device))
            mean_entropy_2 = -torch.mean(
                torch.sum(out_soft_2 * torch.log(out_soft_2 + 1e-10), dim=1)
            ) / normalization
            out_2_result = torch.cat([out_unl_2_result, out_l_2_result], dim=0)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_label_2 = torch.max(out_soft_2.detach(), dim=1)[1]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False) 
            out_2_entropy = entropy_from_softmax(out_soft_2)
            # mask_1_2 = torch.where((out_2_entropy >= 1) & (out_2_entropy < 2), torch.tensor(1, device=out_2_entropy.device), torch.tensor(0, device=out_2_entropy.device))
            # mask_2_2 = torch.where((out_2_entropy < 1), torch.tensor(1, device=out_2_entropy.device), torch.tensor(0, device=out_2_entropy.device))
            # out_2_label_init = out_pseudo_2 * mask_2_2 + out_top2_soft_2 * mask_1_2
            # mask_3_2 = torch.where((out_2_entropy >= 2), torch.tensor(1, device=out_2_entropy.device), torch.tensor(0, device=out_2_entropy.device))
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            mean_entropy = (mean_entropy_1 + mean_entropy_2) / 2
            if mean_entropy < min_entropy :
                min_entropy = mean_entropy

            # Model1 & Model2 Cross Pseudo Supervision
            # if iter_num >= 15000:
            #     _, mask_en_ul_1 = generate_masks(out_max_1, out_1_entropy,B1)
            #     _, mask_en_ul_2 = generate_masks(out_max_2, out_2_entropy,B1)
            #     mask_en = (mask_en_ul_1 * mask_en_ul_2).bool()
            #     net_input_en_1 = net_input_1 * mask_en.unsqueeze(1)
            #     net_input_en_2 = net_input_2 * mask_en.unsqueeze(1)
            #     net_input_en = torch.cat([net_input_en_1[:B2], net_input_en_2[:B2]], dim=0)
            #     out_en_1,_ = model_1(net_input_en)
            #     out_en_1_soft = torch.softmax(out_en_1, dim=1)
            #     out_en_pseudo_1 = torch.argmax(out_en_1_soft.detach(), dim=1, keepdim=False)

            #     out_en_2,_ = model_2(net_input_en)
            #     out_en_2_soft = torch.softmax(out_en_2, dim=1)
            #     out_en_pseudo_2 = torch.argmax(out_en_2_soft.detach(), dim=1, keepdim=False)

            #     pseudo_supervision1_en = dice_loss(out_en_1_soft, out_en_pseudo_2.unsqueeze(1),mask = mask_en.unsqueeze(1))  
            #     pseudo_supervision2_en = dice_loss(out_en_2_soft, out_en_pseudo_1.unsqueeze(1), mask = mask_en.unsqueeze(1)) 
            # else:
                # pseudo_supervision1_en = 0
                # pseudo_supervision2_en = 0

            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))  
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1)) 
            # pseudo_supervision1_top2 = dice_loss(out_soft_1, out_2_label_init.unsqueeze(1).float(), mask=mask_3.unsqueeze(1))  
            # pseudo_supervision2_top2 = dice_loss(out_soft_2, out_1_label_init.unsqueeze(1).float(), mask=mask_3_2.unsqueeze(1))  
            # if model_1_stop == 100 or model_2_stop == 100:
            #     p2 = (p2 + 1) % 3
            p = iter_num/max_iterations
            # if p <  1/5 :
            #     p2 = 0
            # elif 1/5 <= p < 2/5:
            #     p2 = 1
            # # elif 0.40 <= p < 0.60:
            # #     p2 = 2
            # # elif 0.60 <= p < 0.80:
            # #     p2 = 3
            # elif 2/5 <= p :
            #     p2 = 2
            # ABD-R New Training Sample
            # image_patch_last = ABD_I_Conf(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)
            # image_patch_last = Overlap(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, p, args)
            # image_patch_last = layer(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args, p_small)
            # image_patch_last = layer(out_max_1, out_max_2, out_label_1, out_label_2, net_input_1, net_input_2, out_1, out_2, p, p2, args, p_small)
            # if p <  3/5 :
            #     image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1_result, out_2_result,args)
            # else:
            #     image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)
            image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1_result, out_2_result,args)
            # image_patch_last = hhh(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)

            # print(out_max_1.shape)
            # image_patch_last = Ncut(out_max_1, out_max_2, net_input_1, net_input_2, p, args)
            B3, _, _ = image_patch_last.size()
            image_output_1,_ = model_1(image_patch_last.unsqueeze(1))  
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            # pseduo_mask_1 = (normalize(image_output_1) > (1-p**2)).float()
            # pseduo_masknew_1 = pseduo_mask_1 * image_output_1
            image_output_max_1 = torch.max(image_output_soft_1.detach(), dim=1)[0]
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_out_top2_soft_1 = pseudo_variance_gate(image_output_1)
            image_out_1_entropy = entropy_from_softmax(image_output_soft_1)
            image_output_2,_ = model_2(image_patch_last.unsqueeze(1))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            image_output_max_2 = torch.max(image_output_soft_2.detach(), dim=1)[0]
            # pseduo_mask_2 = (normalize(image_output_2) > (1-p**2)).float()
            # pseduo_masknew_2 = pseduo_mask_2 * image_output_2
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            image_out_top2_soft_2 = pseudo_variance_gate(image_output_2)
            image_out_2_entropy = entropy_from_softmax(image_output_soft_2)

            # if  000 <= iter_num <= 80000:
            #     _, image_mask_en_ul_1 = generate_masks(image_output_max_1, image_out_1_entropy,B3, k=0.2)
            #     _, image_mask_en_ul_2 = generate_masks(image_output_max_2, image_out_2_entropy,B3, k=0.2)
            #     image_mask_en = (image_mask_en_ul_1 * image_mask_en_ul_2).bool()
            #     image_patch_last_en = image_patch_last * image_mask_en
            #     image_out_en_1,_ = model_1(image_patch_last_en.unsqueeze(1))
            #     image_out_en_1_soft = torch.softmax(image_out_en_1, dim=1)
            #     image_out_en_pseudo_1 = torch.argmax(image_out_en_1_soft.detach(), dim=1, keepdim=False)

            #     image_out_en_2,_ = model_2(image_patch_last_en.unsqueeze(1))
            #     image_out_en_2_soft = torch.softmax(image_out_en_2, dim=1)
            #     image_out_en_pseudo_2 = torch.argmax(image_out_en_2_soft.detach(), dim=1, keepdim=False)

            #     pseudo_supervision1_en_last = dice_loss(image_out_en_1_soft, image_out_en_pseudo_2.unsqueeze(1),mask = image_mask_en.unsqueeze(1))  
            #     pseudo_supervision2_en_last = dice_loss(image_out_en_2_soft, image_out_en_pseudo_1.unsqueeze(1), mask = image_mask_en.unsqueeze(1)) 
            # else:
            #     pseudo_supervision1_en_last = 0
            #     pseudo_supervision2_en_last = 0
            # Model1 & Model2 Second Step Cross Pseudo Supervision

            # start_x = 128 -  p_small[p2]
            # start_y = 128 -  p_small[p2]
            # end_x = 128 +  p_small[p2]
            # end_y = 128 +  p_small[p2]
            
            # device = image_output_soft_1.device
            
            # r = int(p_small[p2])
            # y = torch.arange(-r, r, dtype=torch.float32).view(-1, 1)  # 列向量 [2r, 1]
            # x = torch.arange(-r, r, dtype=torch.float32).view(1, -1)  # 行向量 [1, 2r]
            # circle_mask = (x.pow(2) + y.pow(2)) <= r**2
            # mask_4d = circle_mask.unsqueeze(0).unsqueeze(0)  # 形狀 [1,1,2r,2r]
            # mask_3d = circle_mask.unsqueeze(0)               # 形狀 [1,2r,2r]
            # mask_4d = mask_4d.to(device)
            # mask_3d = mask_3d.to(device)
            # region_1 = image_output_soft_1[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            # region_2 = image_output_soft_2[:, :, start_y:end_y, start_x:end_x] * mask_4d.cuda()
            # pseudo_region_1 = pseudo_image_output_1[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()
            # pseudo_region_2 = pseudo_image_output_2[:, start_y:end_y, start_x:end_x] * mask_3d.cuda()

            # pseudo_supervision6 = dice_loss(region_1, pseudo_region_2.unsqueeze(1))
            # pseudo_supervision5 = dice_loss(region_2, pseudo_region_1.unsqueeze(1))
            
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4 
            # loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3  + pseudo_supervision1_en_last
            # loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4  + pseudo_supervision2_en_last
            
            loss = loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer_ema.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()
            # optimizer_ema.step()

            iter_num += 1
            if iter_num % 2 ==0:
                update_ema_variables(model_2, ema_model, 0.99, iter_num)
            else:
                update_ema_variables(model_1, ema_model, 0.99, iter_num)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/model1_loss', loss_1, iter_num)
            writer.add_scalar('info/model2_loss', loss_2, iter_num)
            writer.add_scalar('info/model1/mix_dice', loss_dice_1, iter_num)
            writer.add_scalar('info/model1/mix_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/model2/mix_dice', loss_dice_2, iter_num)
            writer.add_scalar('info/model2/mix_ce', loss_ce_2, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            # logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f' % (iter_num, loss, loss_1, loss_2))
            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f, min_entropy: %f' % (iter_num, loss, loss_1, loss_2, min_entropy))

            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)

                if performance1 > best_performance1:
                    model_1_stop = 0
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_mode_path)
                    torch.save(model_1.state_dict(), save_best_path)
                    # save_net_opt(model_1, optimizer1, save_mode_path)
                    # save_net_opt(model_1,optimizer1, save_best_path)
                else:
                    model_1_stop += 1
                logging.info('iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_result(sampled_batch["image"], sampled_batch["label"], model_2,classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)

                if performance2 > best_performance2:
                    model_2_stop = 0
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_mode_path)
                    torch.save(model_2.state_dict(), save_best_path)
                    # save_net_opt(model_2, optimizer2, save_mode_path)
                    # save_net_opt(model_2,optimizer2, save_best_path)
                else:
                    model_2_stop += 1
                logging.info('iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model_2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/ACDC_layer_weight1_test_7_labeled/pre_train".format(args.exp, args.labelnum)
    self_first_snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/ACDC_{}_{}_labeled/self_first_train".format(args.exp, args.labelnum)
    self_second_snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/ACDC_{}_{}_labeled/self_second_train".format(args.exp, args.labelnum)
    self_third_snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/ACDC_{}_{}_labeled/self_third_train".format(args.exp, args.labelnum)
    self_fourth_snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/ACDC_{}_{}_labeled/self_fourth_train".format(args.exp, args.labelnum)
    self_last_snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/ACDC_{}_{}_labeled/self_last_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_first_snapshot_path,self_second_snapshot_path, self_third_snapshot_path, self_fourth_snapshot_path, self_last_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('/home/v1-4080s/hhy/ABD/code/train_ACDC_BCP.py', self_first_snapshot_path)

    # Pre_train
    # logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # pre_train(args, pre_snapshot_path)

    #Self_train
    logging.basicConfig(filename=self_first_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_first_train(args, pre_snapshot_path, self_first_snapshot_path)


    logging.basicConfig(filename=self_first_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_second_train(args, self_first_snapshot_path, self_second_snapshot_path)
    # # # self_first_train(args, self_first_snapshot_path, self_second_snapshot_path)


    logging.basicConfig(filename=self_first_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_third_train(args, self_second_snapshot_path, self_third_snapshot_path)
    # # self_first_train(args, self_second_snapshot_path, self_third_snapshot_path)


    logging.basicConfig(filename=self_first_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_fourth_train(args, self_third_snapshot_path, self_fourth_snapshot_path)
    # self_first_train(args, self_third_snapshot_path, self_last_snapshot_path)


    logging.basicConfig(filename=self_first_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_last_train(args, self_fourth_snapshot_path, self_last_snapshot_path)
    # self_first_train(args, self_third_snapshot_path, self_last_snapshot_path)


    


