import torch
from einops import rearrange
import random
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np

def ABD_I(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args):
    # ABD-I Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # torch.Size([8, 16])
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    for i in range(args.labeled_bs):
        kl_similarities_1 = torch.empty(args.top_num)
        kl_similarities_2 = torch.empty(args.top_num)
        b = torch.argmin(patches_mean_1[i].detach(), dim=0)
        d = torch.argmin(patches_mean_2[i].detach(), dim=0)
        patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
        patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])
        patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
        patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

        for j in range(args.top_num):
            kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
            kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

        a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
        c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)
        a_ori = patches_mean_1_top4_indices[i, a]
        c_ori = patches_mean_2_top4_indices[i, c]

        max_patch_1 = image_patch_2[i][c_ori]  
        image_patch_1[i][b] = max_patch_1  
        max_patch_2 = image_patch_1[i][a_ori]
        image_patch_2[i][d] = max_patch_2 

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size) 
    return image_patch_last
# def ABD_I_Conf(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args):
#     # ABD-I Bidirectional Displacement Patch
#     patches_1 = rearrange(outputs1_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
#     patches_2 = rearrange(outputs2_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) ##修改的地方
#     image_patch_1 = rearrange(volume_batch.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
#     image_patch_2 = rearrange(volume_batch_strong.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    
#     patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # Original mean calculation
#     patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

#     # Calculate adjacent confidence differences
#     confidence_diff_1 = calculate_adjacent_confidence_differences(patches_mean_1)
#     confidence_diff_2 = calculate_adjacent_confidence_differences(patches_mean_2)

#     patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
#     patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
#     patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)
#     patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)
#     confidence_diff_outputs_1 = calculate_adjacent_confidence_differences(patches_mean_outputs_1)
#     confidence_diff_outputs_2 = calculate_adjacent_confidence_differences(patches_mean_outputs_2)

#     # Use confidence_diff_1 and confidence_diff_2 for top-k sorting instead of original means
#     patches_mean_1_top4_values, patches_mean_1_top4_indices = confidence_diff_1.topk(args.top_num, dim=1)
#     patches_mean_2_top4_values, patches_mean_2_top4_indices = confidence_diff_2.topk(args.top_num, dim=1)

#     for i in range(args.labeled_bs):
#         kl_similarities_1 = torch.empty(args.top_num)
#         kl_similarities_2 = torch.empty(args.top_num)
#         # print("confidence_diff_2.shape:", confidence_diff_2.shape)
#         b = torch.argmin(confidence_diff_1[i].detach(), dim=0)
#         d = torch.argmin(confidence_diff_2[i].detach(), dim=0)
#         # 修改这个部分的索引，使其与 confidence_diff_outputs_1 的维度一致
#         patches_mean_outputs_min_1 = confidence_diff_outputs_1[i, b]  # 移除第三个索引
#         patches_mean_outputs_min_2 = confidence_diff_outputs_2[i, d]  # 移除第三个索引
#         patches_mean_outputs_top4_1 = confidence_diff_outputs_1[i, patches_mean_1_top4_indices[i, :]]  # 无需第三个索引
#         patches_mean_outputs_top4_2 = confidence_diff_outputs_2[i, patches_mean_2_top4_indices[i, :]]  # 无需第三个索引

#         for j in range(args.top_num):
#             kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
#             kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

#         a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
#         c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)
#         a_ori = patches_mean_1_top4_indices[i, a]
#         c_ori = patches_mean_2_top4_indices[i, c]

#         max_patch_1 = image_patch_2[i][c_ori]  
#         image_patch_1[i][b] = max_patch_1  
#         max_patch_2 = image_patch_1[i][a_ori]
#         image_patch_2[i][d] = max_patch_2 

#     image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
#     image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size) 
#     return image_patch_last

def ABD_I_mIoU(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args):
    # ABD-I Bidirectional Displacement Patch with mIoU
    # Rearrange patches from outputs and volumes
    patches_1_max = rearrange(outputs1_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2_max = rearrange(outputs2_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_1_unlabel = rearrange(outputs1_unlabel, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2_unlabel = rearrange(outputs2_unlabel, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    
    image_patch_1 = rearrange(volume_batch.squeeze(1)[args.labeled_bs:], 'b (h p1) (w p2) -> b (h w)(p1 p2)', p1=args.patch_size, p2=args.patch_size)  
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1)[args.labeled_bs:], 'b (h p1) (w p2) -> b (h w)(p1 p2)', p1=args.patch_size, p2=args.patch_size)
    
    patches_mIoU_1 = []
    patches_mIoU_2 = []

    # 计算outputs1_max和outputs2_unlabel的mIoU1
    for i in range(patches_1_max.shape[1]):
        intersection_1 = torch.sum((patches_1_max[:, i] * patches_2_unlabel[:, i]), dim=1)
        union_1 = torch.sum(patches_1_max[:, i] + patches_2_unlabel[:, i] - patches_1_max[:, i] * patches_2_unlabel[:, i], dim=1)
        mIoU_1 = intersection_1 / (union_1 + 1e-6)  # mIoU for each patch in image 1 and 2
        patches_mIoU_1.append(mIoU_1)

    # 计算outputs2_max和outputs1_unlabel的mIoU2
    for i in range(patches_2_max.shape[1]):
        intersection_2 = torch.sum((patches_2_max[:, i] * patches_1_unlabel[:, i]), dim=1)
        union_2 = torch.sum(patches_2_max[:, i] + patches_1_unlabel[:, i] - patches_2_max[:, i] * patches_1_unlabel[:, i], dim=1)
        mIoU_2 = intersection_2 / (union_2 + 1e-6)  # mIoU for each patch in image 2 and 1
        patches_mIoU_2.append(mIoU_2)

    # Convert to tensors for easier handling
    patches_mIoU_1 = torch.stack(patches_mIoU_1, dim=1)
    patches_mIoU_2 = torch.stack(patches_mIoU_2, dim=1)

    # 取出mIoU1最小块和mIoU2最大块
    lowest_mIoU1_indices = torch.argmin(patches_mIoU_1, dim=1)
    highest_mIoU2_indices = torch.argmax(patches_mIoU_2, dim=1)

    # 取出mIoU2最小块和mIoU1最大块
    lowest_mIoU2_indices = torch.argmin(patches_mIoU_2, dim=1)
    highest_mIoU1_indices = torch.argmax(patches_mIoU_1, dim=1)

    # 交换patch
    for i in range(args.labeled_bs):
        # 用mIoU2最大的块替换mIoU1最小的块
        high_patch_2 = image_patch_2[i][highest_mIoU2_indices[i]]
        image_patch_1[i][lowest_mIoU1_indices[i]] = high_patch_2

        # 用mIoU1最大的块替换mIoU2最小的块
        high_patch_1 = image_patch_1[i][highest_mIoU1_indices[i]]
        image_patch_2[i][lowest_mIoU2_indices[i]] = high_patch_1

    # 拼接替换后的patches
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)

    # 将patches恢复到原始图像大小
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)

    return image_patch_last

def ABD_I_Conf(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args):
    # ABD-I Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) ##修改的地方
    image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # Original mean calculation
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # Calculate adjacent confidence differences
    # print(patches_mean_1.shape)
    confidence_diff_1 = calculate_adjacent_confidence_differences(patches_mean_1)
    confidence_diff_2 = calculate_adjacent_confidence_differences(patches_mean_2)
    # print(confidence_diff_1.shape)

    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)
    # confidence_diff_outputs_1 = calculate_adjacent_confidence_differences(patches_mean_outputs_1)
    # confidence_diff_outputs_2 = calculate_adjacent_confidence_differences(patches_mean_outputs_2)

    # Use confidence_diff_1 and confidence_diff_2 for top-k sorting instead of original means
    patches_mean_1_top4_values, patches_mean_1_top4_indices = confidence_diff_1.topk(args.top_num, dim=1, largest=False)
    patches_mean_2_top4_values, patches_mean_2_top4_indices = confidence_diff_2.topk(args.top_num, dim=1, largest=False)

    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)
            # print("confidence_diff_2.shape:", confidence_diff_2.shape)
            b = torch.argmax(confidence_diff_1[i].detach(), dim=0)
            d = torch.argmax(confidence_diff_2[i].detach(), dim=0)
            # 修改这个部分的索引，使其与 confidence_diff_outputs_1 的维度一致
            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b]  # 移除第三个索引
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d]  # 移除第三个索引
            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :]]  # 无需第三个索引
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :]]  # 无需第三个索引

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
                kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)
            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

            max_patch_1 = image_patch_2[i][d]  
            image_patch_1[i][a_ori] = max_patch_1  
            max_patch_2 = image_patch_1[i][b]
            image_patch_2[i][c_ori] = max_patch_2 
        else:
            hardest_patch_1 = torch.argmax(confidence_diff_1, dim=1)
            easiest_patch_1 = torch.argmin(confidence_diff_1, dim=1)
            hardest_patch_2 = torch.argmax(confidence_diff_2, dim=1)
            easiest_patch_2 = torch.argmin(confidence_diff_2, dim=1)

            # 将第二组中的最难块放入第一组中最易块的位置
            max_patch_1 = image_patch_2[i][hardest_patch_2]  
            image_patch_1[i][easiest_patch_1] = max_patch_1  
            max_patch_2 = image_patch_1[i][hardest_patch_1]
            image_patch_2[i][easiest_patch_2] = max_patch_2 

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size) 
    return image_patch_last

def ABD_I_BCP(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args):
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ',p1=args.patch_size, p2=args.patch_size)  # torch.Size([12, 224, 224])
    image_patch_2 = rearrange(net_input_2.squeeze(1),'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(out_1, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(out_2, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])

    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)
            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])

            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
                kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

            max_patch_1 = image_patch_2[i][c_ori]  
            image_patch_1[i][b] = max_patch_1  
            max_patch_2 = image_patch_1[i][a_ori]
            image_patch_2[i][d] = max_patch_2
        else:
            a = torch.argmax(patches_mean_1[i].detach(), dim=0)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            c = torch.argmax(patches_mean_2[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)

            max_patch_1 = image_patch_2[i][c]  
            image_patch_1[i][b] = max_patch_1  
            max_patch_2 = image_patch_1[i][a]
            image_patch_2[i][d] = max_patch_2
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([24, 224, 224])
    return image_patch_last

def ABD_R(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args):
    # ABD-R Bidirectional Displacement Patch
    patches_supervised_1 = rearrange(outputs1_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_1 = rearrange(volume_batch.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_supervised_2 = rearrange(volume_batch_strong.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_1 = rearrange(label_batch[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_2 = rearrange(label_batch_strong[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_supervised_1 = torch.mean(patches_supervised_1.detach(), dim=2)
    patches_mean_supervised_2 = torch.mean(patches_supervised_2.detach(), dim=2)
    e = torch.argmax(patches_mean_supervised_1.detach(), dim=1)
    f = torch.argmin(patches_mean_supervised_1.detach(), dim=1)
    g = torch.argmax(patches_mean_supervised_2.detach(), dim=1)
    h = torch.argmin(patches_mean_supervised_2.detach(), dim=1)
    # print(g)
    # print(h)
    for i in range(args.labeled_bs): 
        if random.random() < 0.5:
            min_patch_supervised_1 = image_patch_supervised_2[i][h[i]]
            image_patch_supervised_1[i][e[i]] = min_patch_supervised_1
            min_patch_supervised_2 = image_patch_supervised_1[i][f[i]]
            image_patch_supervised_2[i][g[i]] = min_patch_supervised_2

            min_label_supervised_1 = label_patch_supervised_2[i][h[i]]
            label_patch_supervised_1[i][e[i]] = min_label_supervised_1
            min_label_supervised_2 = label_patch_supervised_1[i][f[i]]
            label_patch_supervised_2[i][g[i]] = min_label_supervised_2
    image_patch_supervised = torch.cat([image_patch_supervised_1, image_patch_supervised_2], dim=0)
    image_patch_supervised_last = rearrange(image_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    label_patch_supervised = torch.cat([label_patch_supervised_1, label_patch_supervised_2], dim=0)
    label_patch_supervised_last = rearrange(label_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    return image_patch_supervised_last, label_patch_supervised_last

def ABD_R_BS(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, args):
    # ABD-R Bidirectional Displacement Patch
    patches_supervised_1_U = rearrange(outputs1_max[args.labeled_bs // 2:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2_U = rearrange(outputs2_max[args.labeled_bs // 2:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    image_patch_supervised_1_L = F.avg_pool2d(
        volume_batch[:args.labeled_bs //2 ],  # 增加 channel 维度
        kernel_size=4,
        stride=4
    ).squeeze(1)  # 恢复到没有 channel 的形状
    image_patch_supervised_2_L = F.avg_pool2d(
        volume_batch_strong[:args.labeled_bs // 2],  # 增加 channel 维度
        kernel_size=4,
        stride=4
    ).squeeze(1)  # 恢复到没有 channel 的形状
    image_patch_supervised_1_L = rearrange(image_patch_supervised_1_L, 'b p1 p2 -> b (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_2_L = rearrange(image_patch_supervised_2_L, 'b p1 p2 -> b (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_1_U = rearrange(volume_batch.squeeze(1)[args.labeled_bs // 2:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_supervised_2_U = rearrange(volume_batch_strong.squeeze(1)[args.labeled_bs // 2:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    # label_patch_supervised_1_L = F.avg_pool2d(
    #     label_batch[:args.labeled_bs].unsqueeze(1),  # 增加 channel 维度
    #     kernel_size=args.patch_size,
    #     stride=args.patch_size
    # ).squeeze(1)  # 恢复到没有 channel 的形状
    # label_patch_supervised_2_L = F.avg_pool2d(
    #     label_batch_strong[:args.labeled_bs].unsqueeze(1),  # 增加 channel 维度
    #     kernel_size=args.patch_size,
    #     stride=args.patch_size
    # ).squeeze(1)  # 恢复到没有 channel 的形状
    # label_patch_supervised_1_U = rearrange(outputs2_unlabel, 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    # label_patch_supervised_2_U = rearrange(outputs1_unlabel, 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_supervised_1 = torch.mean(patches_supervised_1_U.detach(), dim=2)
    patches_mean_supervised_2 = torch.mean(patches_supervised_2_U.detach(), dim=2)
    # 初始化 KL 相似度，设置为一个较大的初始值（例如无穷大）
    kl_similarities_1 = torch.full((args.labeled_bs // 2,), float('inf'))  # 初始为正无穷大
    kl_similarities_2 = torch.full((args.labeled_bs // 2,), float('inf'))  # 初始为正无穷大

    # 初始化最小 KL 对应的索引
    kl_indices_1 = torch.zeros(args.labeled_bs // 2, dtype=torch.long)
    kl_indices_2 = torch.zeros(args.labeled_bs // 2, dtype=torch.long)

    # 遍历计算 KL 相似度
    for b in range(args.labeled_bs // 2):
        for j in range(args.labeled_bs // 2):
            # 计算双向 KL 散度
            kl_1 = torch.nn.functional.kl_div(
                outputs1_max[args.labeled_bs // 2 + b].log(),
                outputs2_max[j],
                reduction='sum'
            )
            kl_2 = torch.nn.functional.kl_div(
                outputs2_max[args.labeled_bs // 2 + b].log(),
                outputs1_max[j],
                reduction='sum'
            )

            # 更新最小 KL 和对应索引
            if kl_1 < kl_similarities_1[b]:
                kl_similarities_1[b] = kl_1
                kl_indices_1[b] = j  # 更新对应的索引
            
            if kl_2 < kl_similarities_2[b]:
                kl_similarities_2[b] = kl_2
                kl_indices_2[b] = j  # 更新对应的索引

    e = torch.argmax(patches_mean_supervised_1.detach(), dim=1)
    f = torch.argmin(patches_mean_supervised_1.detach(), dim=1)
    g = torch.argmax(patches_mean_supervised_2.detach(), dim=1)
    h = torch.argmin(patches_mean_supervised_2.detach(), dim=1)

    for i in range(args.labeled_bs // 2): 
        if random.random() < 0.5:
            min_patch_supervised_1 = image_patch_supervised_2_L[kl_indices_2[i]]
            image_patch_supervised_1_U[i][f[i]] = min_patch_supervised_1
            min_patch_supervised_2 = image_patch_supervised_1_L[kl_indices_1[i]]
            image_patch_supervised_2_U[i][h[i]] = min_patch_supervised_2

            # min_label_supervised_1 = label_patch_supervised_2_L
            # label_patch_supervised_1_U[i][f[i]] = min_label_supervised_1
            # min_label_supervised_2 = label_patch_supervised_1_L
            # label_patch_supervised_2_U[i][f[i]] = min_label_supervised_2
    image_patch_supervised = torch.cat([image_patch_supervised_1_U, image_patch_supervised_2_U], dim=0)
    image_patch_supervised_last = rearrange(image_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    
    return image_patch_supervised_last

def calculate_adjacent_confidence_differences(patches_mean):
    if patches_mean.dim() == 3:  # If patches_mean has 3 dimensions (b, n, c)
        patches_mean = torch.mean(patches_mean, dim=2)  # Take mean across the channel dimension

    b, n = patches_mean.shape  # b: batch size, n: number of patches (h * w)
    h = w = int(n**0.5)  # Assuming patches are arranged in a square grid
    confidence_diff = torch.zeros_like(patches_mean)

    # For each patch, calculate the difference with its adjacent patches
    for i in range(h):
        for j in range(w):
            idx = i * w + j  # Current patch index
            neighbors = []
            if i > 0:  # Top neighbor
                neighbors.append(patches_mean[:, (i-1) * w + j])
            if i < h - 1:  # Bottom neighbor
                neighbors.append(patches_mean[:, (i+1) * w + j])
            if j > 0:  # Left neighbor
                neighbors.append(patches_mean[:, i * w + (j-1)])
            if j < w - 1:  # Right neighbor
                neighbors.append(patches_mean[:, i * w + (j+1)])
            if neighbors:
                # Compute the mean of the differences
                neighbor_diff = [torch.abs(patches_mean[:, idx] - neighbor) for neighbor in neighbors]
                confidence_diff[:, idx] = torch.mean(torch.stack(neighbor_diff, dim=0), dim=0)
    return confidence_diff

def ABD_R_Conf(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args):
    # ABD-R Bidirectional Displacement Patch
    patches_supervised_1 = rearrange(outputs1_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_1 = rearrange(volume_batch.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_supervised_2 = rearrange(volume_batch_strong.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_1 = rearrange(label_batch[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_2 = rearrange(label_batch_strong[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_supervised_1 = torch.mean(patches_supervised_1.detach(), dim=2)
    patches_mean_supervised_2 = torch.mean(patches_supervised_2.detach(), dim=2)

    # 计算每个图像块与其相邻块之间的置信度差异
    confidence_diff_1 = calculate_adjacent_confidence_differences(patches_mean_supervised_1)
    confidence_diff_2 = calculate_adjacent_confidence_differences(patches_mean_supervised_2)

    # 选择置信度差异最大的块（难学习的区域）和差异最小的块（易学习的区域）
    hardest_patch_1 = torch.argmax(confidence_diff_1, dim=1)
    easiest_patch_1 = torch.argmin(confidence_diff_1, dim=1)
    hardest_patch_2 = torch.argmax(confidence_diff_2, dim=1)
    easiest_patch_2 = torch.argmin(confidence_diff_2, dim=1)

    # 将难学习的块放入易学习区域
    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            # 将第二组中的最难块放入第一组中最易块的位置
            hard_patches_1 = image_patch_supervised_2[i][hardest_patch_2[i]]
            hard_patches_2 = image_patch_supervised_1[i][hardest_patch_1[i]]
            hard_label_1 = label_patch_supervised_2[i][hardest_patch_2[i]]
            hard_label_2 = label_patch_supervised_1[i][hardest_patch_1[i]]

            image_patch_supervised_1[i][easiest_patch_1[i]] = hard_patches_1
            label_patch_supervised_1[i][easiest_patch_1[i]] = hard_label_1
            
            # 将第一组中的最难块放入第二组中最易块的位置
            
            image_patch_supervised_2[i][easiest_patch_2[i]] = hard_patches_2
            label_patch_supervised_2[i][easiest_patch_2[i]] = hard_label_2

    image_patch_supervised = torch.cat([image_patch_supervised_1, image_patch_supervised_2], dim=0)
    image_patch_supervised_last = rearrange(image_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    label_patch_supervised = torch.cat([label_patch_supervised_1, label_patch_supervised_2], dim=0)
    label_patch_supervised_last = rearrange(label_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    return image_patch_supervised_last, label_patch_supervised_last

def calculate_adjacent_confidence_differences_3d(patches_mean):
    if patches_mean.dim() == 3:  # If patches_mean has 3 dimensions (b, n, c)
        patches_mean = torch.mean(patches_mean, dim=2)  # Take mean across the channel dimension

    b, n = patches_mean.shape  # b: batch size, n: number of patches (h * w)
    h = w = d = int(n**(1/3))  # Assuming patches are arranged in a square grid
    confidence_diff = torch.zeros_like(patches_mean)

    # For each patch, calculate the difference with its adjacent patches
    for i in range(h):
        for j in range(w):
            idx = i * w + j  # Current patch index
            neighbors = []
            if i > 0:  # Top neighbor
                neighbors.append(patches_mean[:, (i-1) * w + j])
            if i < h - 1:  # Bottom neighbor
                neighbors.append(patches_mean[:, (i+1) * w + j])
            if j > 0:  # Left neighbor
                neighbors.append(patches_mean[:, i * w + (j-1)])
            if j < w - 1:  # Right neighbor
                neighbors.append(patches_mean[:, i * w + (j+1)])
            if neighbors:
                # Compute the mean of the differences
                neighbor_diff = [torch.abs(patches_mean[:, idx] - neighbor) for neighbor in neighbors]
                confidence_diff[:, idx] = torch.mean(torch.stack(neighbor_diff, dim=0), dim=0)
    return confidence_diff

def BD_S(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args):
    # 按块划分预测值、图像和标签
    patches_supervised_1 = rearrange(outputs1_max[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_1 = rearrange(volume_batch.squeeze(1)[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_2 = rearrange(volume_batch_strong.squeeze(1)[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_1 = rearrange(label_batch[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_2 = rearrange(label_batch_strong[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    
    # 计算每个块的置信度均值
    patches_mean_supervised_1 = torch.mean(patches_supervised_1, dim=2)
    patches_mean_supervised_2 = torch.mean(patches_supervised_2, dim=2)
    
    # 根据置信度进行排序
    _, indices_asc_1 = torch.sort(patches_mean_supervised_1, dim=1, descending=False)
    _, indices_desc_1 = torch.sort(patches_mean_supervised_1, dim=1, descending=True)
    _, indices_asc_2 = torch.sort(patches_mean_supervised_2, dim=1, descending=False)
    _, indices_desc_2 = torch.sort(patches_mean_supervised_2, dim=1, descending=True)

    reordered_image_patch_asc_1 = image_patch_supervised_1
    reordered_label_patch_asc_1 = label_patch_supervised_1
    reordered_image_patch_desc_1 = image_patch_supervised_1
    reordered_label_patch_desc_1 = label_patch_supervised_1
    
    reordered_image_patch_asc_2 = image_patch_supervised_2
    reordered_label_patch_asc_2 = label_patch_supervised_2
    reordered_image_patch_desc_2 = image_patch_supervised_2
    reordered_label_patch_desc_2 = label_patch_supervised_2
    
    if random.random() < 0.5:
        # 根据排序重排图像块和标签块
        reordered_image_patch_asc_1 = torch.stack([image_patch_supervised_1[i][indices_asc_1[i]] for i in range(args.labeled_bs)])
        reordered_label_patch_asc_1 = torch.stack([label_patch_supervised_1[i][indices_asc_1[i]] for i in range(args.labeled_bs)])
        reordered_image_patch_desc_1 = torch.stack([image_patch_supervised_1[i][indices_desc_1[i]] for i in range(args.labeled_bs)])
        reordered_label_patch_desc_1 = torch.stack([label_patch_supervised_1[i][indices_desc_1[i]] for i in range(args.labeled_bs)])
        
        reordered_image_patch_asc_2 = torch.stack([image_patch_supervised_2[i][indices_asc_2[i]] for i in range(args.labeled_bs)])
        reordered_label_patch_asc_2 = torch.stack([label_patch_supervised_2[i][indices_asc_2[i]] for i in range(args.labeled_bs)])
        reordered_image_patch_desc_2 = torch.stack([image_patch_supervised_2[i][indices_desc_2[i]] for i in range(args.labeled_bs)])
        reordered_label_patch_desc_2 = torch.stack([label_patch_supervised_2[i][indices_desc_2[i]] for i in range(args.labeled_bs)])
    
    # 重组排序后的图像和标签
    reordered_image_patch = torch.cat([reordered_image_patch_asc_1, reordered_image_patch_asc_2, reordered_image_patch_desc_1, reordered_image_patch_desc_2], dim=0)
    reordered_label_patch = torch.cat([reordered_label_patch_asc_1, reordered_label_patch_asc_2, reordered_label_patch_desc_1, reordered_label_patch_desc_2], dim=0)
    reordered_image_patch_last = rearrange(reordered_image_patch, 'b (h w) (p1 p2) -> b (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    reordered_label_patch_last = rearrange(reordered_label_patch, 'b (h w) (p1 p2) -> b (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    
    return reordered_image_patch_last, reordered_label_patch_last

def BD_S_z(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args):
    # 按块划分预测值、图像和标签
    patches_supervised_1 = rearrange(outputs1_max[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_1 = rearrange(volume_batch.squeeze(1)[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_2 = rearrange(volume_batch_strong.squeeze(1)[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_1 = rearrange(label_batch[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_2 = rearrange(label_batch_strong[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    # 计算每个块的置信度均值
    patches_mean_supervised_1 = torch.mean(patches_supervised_1, dim=2)  # 每个块的平均置信度
    patches_mean_supervised_2 = torch.mean(patches_supervised_2, dim=2)

    # 对所有块的均值进行正态分布分析 (z 分数)
    mean_1 = torch.mean(patches_mean_supervised_1, dim=1, keepdim=True)
    std_1 = torch.std(patches_mean_supervised_1, dim=1, keepdim=True)
    z_scores_1 = (patches_mean_supervised_1 - mean_1) / (std_1 + 1e-6)

    mean_2 = torch.mean(patches_mean_supervised_2, dim=1, keepdim=True)
    std_2 = torch.std(patches_mean_supervised_2, dim=1, keepdim=True)
    z_scores_2 = (patches_mean_supervised_2 - mean_2) / (std_2 + 1e-6)

    # 根据 z 分数从低到高排序
    _, indices_sorted_1 = torch.sort(z_scores_1, dim=1)
    _, indices_sorted_2 = torch.sort(z_scores_2, dim=1)

    reordered_image_patch_1 = image_patch_supervised_1
    reordered_label_patch_1 = label_patch_supervised_1
    reordered_image_patch_2 = image_patch_supervised_2
    reordered_label_patch_2 = label_patch_supervised_2
    
    if random.random() < 0.5:
    # 根据排序重排图像块和标签块
        reordered_image_patch_1 = torch.stack([image_patch_supervised_1[i][indices_sorted_1[i]] for i in range(args.labeled_bs)])
        reordered_label_patch_1 = torch.stack([label_patch_supervised_1[i][indices_sorted_1[i]] for i in range(args.labeled_bs)])
        reordered_image_patch_2 = torch.stack([image_patch_supervised_2[i][indices_sorted_2[i]] for i in range(args.labeled_bs)])
        reordered_label_patch_2 = torch.stack([label_patch_supervised_2[i][indices_sorted_2[i]] for i in range(args.labeled_bs)])

    # 合并重组排序后的图像和标签
    reordered_image_patch = torch.cat([reordered_image_patch_1, reordered_image_patch_2], dim=0)
    reordered_label_patch = torch.cat([reordered_label_patch_1, reordered_label_patch_2], dim=0)
    reordered_image_patch_last = rearrange(reordered_image_patch, 'b (h w) (p1 p2) -> b (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    reordered_label_patch_last = rearrange(reordered_label_patch, 'b (h w) (p1 p2) -> b (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)

    return reordered_image_patch_last, reordered_label_patch_last

def BD_S_Unlabel(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, args):
    # 按块划分预测值、图像和标签
    patches_supervised_1 = rearrange(outputs1_max[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_1 = rearrange(volume_batch.squeeze(1)[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_2 = rearrange(volume_batch_strong.squeeze(1)[:args.labeled_bs], 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    
    # 计算每个块的置信度均值
    patches_mean_supervised_1 = torch.mean(patches_supervised_1, dim=2)
    patches_mean_supervised_2 = torch.mean(patches_supervised_2, dim=2)
    
    # 根据置信度进行排序
    _, indices_asc_1 = torch.sort(patches_mean_supervised_1, dim=1, descending=False)
    _, indices_desc_1 = torch.sort(patches_mean_supervised_1, dim=1, descending=True)
    _, indices_asc_2 = torch.sort(patches_mean_supervised_2, dim=1, descending=False)
    _, indices_desc_2 = torch.sort(patches_mean_supervised_2, dim=1, descending=True)
    
    # 根据排序重排图像块和标签块
    reordered_image_patch_asc_1 = torch.stack([image_patch_supervised_1[i][indices_asc_1[i]] for i in range(args.labeled_bs)])
    # reordered_image_patch_desc_1 = torch.stack([image_patch_supervised_1[i][indices_desc_1[i]] for i in range(args.labeled_bs)])
    reordered_image_patch_asc_2 = torch.stack([image_patch_supervised_2[i][indices_asc_2[i]] for i in range(args.labeled_bs)])
    # reordered_image_patch_desc_2 = torch.stack([image_patch_supervised_2[i][indices_desc_2[i]] for i in range(args.labeled_bs)])

    # 拼接前半段和后半段
    half_size = patches_mean_supervised_1.shape[1] // 2

    kl_divergences = torch.zeros(args.labeled_bs, args.labeled_bs)  # KL散度矩阵
    for i in range(args.labeled_bs):
        for j in range(args.labeled_bs):
            part1 = patches_mean_supervised_1[i, :half_size].log()
            part2 = patches_mean_supervised_2[j, :half_size].log()
            if part1.shape != part2.shape:
                raise ValueError(f"Shape mismatch: {part1.shape} vs {part2.shape}")
            kl_divergences[i, j] = F.kl_div(part1, part2, reduction='batchmean')

    kl_divergences_inver = torch.zeros(args.labeled_bs, args.labeled_bs)  # KL散度矩阵
    for i in range(args.labeled_bs):
        for j in range(args.labeled_bs):
            part1 = patches_mean_supervised_2[i, :half_size].log()
            part2 = patches_mean_supervised_1[j, :half_size].log()
            if part1.shape != part2.shape:
                raise ValueError(f"Shape mismatch: {part1.shape} vs {part2.shape}")
            kl_divergences_inver[i, j] = F.kl_div(part1, part2, reduction='batchmean')

    # 找到KL散度最小的索引
    min_kl_indices = torch.argmin(kl_divergences, dim=1)
    min_kl_indices_inver = torch.argmin(kl_divergences_inver, dim=1)

    # 拼接图像
    concatenated_image_patch = []
    if random.random() < 0.5:
        # print(111111111111111111)
        for i in range(args.labeled_bs):
                # 图像1的前半段
                image_patch_front = reordered_image_patch_asc_1[i, :half_size]
                # 根据KL散度最小的索引，找到图像2中对应的后半段
                corresponding_index = min_kl_indices[i]
                image_patch_back = reordered_image_patch_asc_2[corresponding_index, half_size:]
                concatenated_image_patch.append(torch.cat([image_patch_front, image_patch_back], dim=0))
                
                # print(concatenated_image_patch.shape)

        for i in range(args.labeled_bs):
            # 图像1的前半段
                image_patch_front = reordered_image_patch_asc_2[i, :half_size]
                # 根据KL散度最小的索引，找到图像2中对应的后半段
                corresponding_index = min_kl_indices_inver[i]
                image_patch_back = reordered_image_patch_asc_1[corresponding_index, half_size:]
                concatenated_image_patch.append(torch.cat([image_patch_front, image_patch_back], dim=0))
        
        concatenated_image_patch = torch.stack(concatenated_image_patch)
                
    else:
        # print(22222222222222222222222)
        concatenated_image_patch = torch.cat([reordered_image_patch_asc_1, reordered_image_patch_asc_2], dim=0)
        # print(concatenated_image_patch[0].shape)
        


    # 重组拼接后的图像
    concatenated_image_patch_last = rearrange(concatenated_image_patch, 'b (h w) (p1 p2) -> b (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    
    return concatenated_image_patch_last

def build_adjacency_matrix_from_flat_labels(batch_labels, image_shape=(16, 16), device="cuda"):
    """
    根据平坦化后的分割标签生成邻接矩阵。每个像素点和相邻的像素点如果属于同一类别，则认为相连。
    :param batch_labels: 图像分割结果，shape 为 (B, H*W)，每个像素点的值是所属类别。
    :param image_shape: 图像的尺寸，用于构建邻接矩阵，默认为 (H, W)。
    :param device: 运行的设备 ('cuda' 或 'cpu')。
    :return: 邻接矩阵，shape 为 (B, H*W, H*W)。
    """
    B, num_pixels = batch_labels.shape  # B 是批次大小，num_pixels = H * W
    h, w = image_shape

    # 初始化邻接矩阵，形状为 (B, H*W, H*W)
    adj_matrices = torch.zeros((B, num_pixels, num_pixels), dtype=torch.float32, device=device)

    # 对每一张图片计算邻接矩阵
    for b in range(B):
        labels = batch_labels[b]  # 获取当前批次的标签
        adj_matrix = adj_matrices[b]  # 当前图像的邻接矩阵

        # 遍历每个像素
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                current_label = labels[idx]

                # 检查相邻像素（上下左右）
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    # 确保邻居索引在有效范围内
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_idx = ni * w + nj
                        neighbor_label = labels[neighbor_idx]

                        # 如果邻居属于相同的类别，就连接
                        if abs(current_label - neighbor_label) < 2:
                            adj_matrix[idx, neighbor_idx] = 1.0

    return adj_matrices


# Ncut 算法函数
def ncut(adj_matrix, tau=0.15, eps=1e-5, eig_vecs=2, device='cuda'):
    """
    计算归一化切分（Ncut）算法，用于图像分割。
    :param adj_matrix: 邻接矩阵，shape 为 (B, H*W, H*W)，B 是批量大小，H*W 是每张图像的像素数。
    :param tau: 平滑系数，控制图像分割的细节。
    :param eps: 防止除零错误的小常数。
    :param eig_vecs: 返回的特征向量数量。
    :param device: 运行设备（'cuda' 或 'cpu'）。
    :return: 特征向量和特征值。
    """
    B, num_pixels, _ = adj_matrix.shape  # B 是批次大小，num_pixels = H*W

    eigenvectors = []
    eigenvalues = []

    for b in range(B):
        A = adj_matrix[b]  # 获取当前批次的邻接矩阵，shape 为 (H*W, H*W)
        
        # 计算度矩阵 D
        D = torch.diag(torch.sum(A, dim=1))

        # 添加小常数以防止零度节点导致的奇异矩阵
        D += torch.eye(D.size(0), device=D.device) * eps

        # 归一化邻接矩阵
        D_diag = torch.sqrt(D)
        D_over_sqrt = torch.inverse(D_diag)

        # 计算归一化拉普拉斯矩阵 L = D^(-1/2) * (D - A) * D^(-1/2)
        L = D_over_sqrt @ (D - A) @ D_over_sqrt

        # 求解特征值和特征向量
        eigvals, eigvecs = torch.linalg.eigh(L)

        # 获取前 eig_vecs 个特征向量
        eigvals = eigvals[:eig_vecs]
        eigvecs = eigvecs[:, :eig_vecs]

        eigenvectors.append(eigvecs)
        eigenvalues.append(eigvals)

    # 将结果转化为 Tensor
    eigenvectors = torch.stack(eigenvectors, dim=0)
    eigenvalues = torch.stack(eigenvalues, dim=0)

    return eigenvectors, eigenvalues

def partition_graph(eigenvectors, num_subgraphs=2, percent=0.5):
    """
    根据第二小的特征向量划分图为两个子图。
    :param eigenvectors: 特征向量，shape为(batch_size, num_nodes, eig_vecs)
    :param num_subgraphs: 需要划分的子图数量，默认为2。
    :param percent: 划分子图1的节点比例，默认为50%。
    :return: 划分后的子图节点索引。
    """
    # 选择第二小的特征向量（索引1）
    second_eigenvector = eigenvectors[:, :, 1]
    
    batch_size = eigenvectors.shape[0]
    num_nodes = eigenvectors.shape[1]
    subgraph_1_count = int(num_nodes * percent)
    
    # 如果子图1的节点数小于1，则设置为至少1个节点
    if subgraph_1_count < 1:
        subgraph_1_count = 1
    elif subgraph_1_count > num_nodes:
        subgraph_1_count = num_nodes - 1
    
    subgraph_1 = []
    subgraph_2 = []
    
    # 对于每个batch，进行划分
    for i in range(batch_size):
        # 获取当前batch的第二小特征向量
        eigenvector = second_eigenvector[i]
        
        # 按值排序并选择阈值
        sorted_indices = torch.argsort(eigenvector)
        sorted_eigenvector = eigenvector[sorted_indices]
        threshold_value = sorted_eigenvector[subgraph_1_count]
        
        # 划分成两个子图：大于等于阈值为子图1，小于阈值为子图2
        subgraph_1_indices = (eigenvector >= threshold_value).nonzero(as_tuple=True)[0]
        subgraph_2_indices = (eigenvector < threshold_value).nonzero(as_tuple=True)[0]
        
        subgraph_1.append(subgraph_1_indices)
        subgraph_2.append(subgraph_2_indices)
    
    # 填充到相同大小
    subgraph_1 = rnn_utils.pad_sequence(subgraph_1, batch_first=True, padding_value=-1)
    subgraph_2 = rnn_utils.pad_sequence(subgraph_2, batch_first=True, padding_value=-1)
    
    return subgraph_1, subgraph_2


def Ncut(out_max_1, out_max_2, net_input_1, net_input_2, iter_nums, args):
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size).float()
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size).float()
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_2 = rearrange(net_input_2.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2)', p1=args.patch_size, p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # print(patches_mean_1.shape)
    adj_matrix_1 = build_adjacency_matrix_from_flat_labels(patches_mean_1, image_shape=(4, 4), device='cuda')
    adj_matrix_2 = build_adjacency_matrix_from_flat_labels(patches_mean_2, image_shape=(4, 4), device='cuda')

    eigenvectors_1, eigenvalues_1 = ncut(adj_matrix_1, tau=0.15, eps=1e-5, eig_vecs=2, device='cuda')
    eigenvectors_2, eigenvalues_2 = ncut(adj_matrix_2, tau=0.15, eps=1e-5, eig_vecs=2, device='cuda')

    subgraph_1_1, subgraph_1_2 = partition_graph(eigenvectors_1, percent=iter_nums)
    subgraph_2_1, subgraph_2_2 = partition_graph(eigenvectors_2, percent=iter_nums)

    # print(subgraph_1_1.shape)
    batch_indices_1 = subgraph_1_1[0]
    node_indices_1 = subgraph_1_1[1]
    batch_indices_2 = subgraph_2_1[0]
    node_indices_2 = subgraph_2_1[1]
    # batch_indices_1, node_indices_1 = subgraph_1_1
    # batch_indices_2, node_indices_2 = subgraph_2_1


    # 遍历所有标记数据
    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            # 从 subgraph_1_1 中提取对应的像素/patch
            for j in range(len(batch_indices_1)):
                if batch_indices_1[j] == i:
                    max_patch_1 = image_patch_2[i][node_indices_1[j]]  # 获取属于 subgraph_1_1 的图像patch
                    image_patch_1[i][node_indices_1[j]] = max_patch_1  # 将其复制到 image_patch_1 上
            
            for z in range(len(batch_indices_2)):
                if batch_indices_1[z] == i:
                    max_patch_2 = image_patch_1[i][node_indices_2[z]]  # 获取属于 subgraph_1_1 的图像patch
                    image_patch_2[i][node_indices_2[z]] = max_patch_2  # 将其复制到 image_patch_1 上

        else:
            # 遍历每个批次
            for i in range(args.labeled_bs):
                # 遍历batch_indices_1中的每个元素，检查它是否属于当前批次
                for j in range(len(batch_indices_1)):
                    if batch_indices_1[j] == i:  # 当前批次
                        # 在batch_indices_2中查找与node_indices_1[j]对应的元素
                        for z in range(len(batch_indices_2)):
                            if batch_indices_2[z] == i and node_indices_2[z] == node_indices_1[j]:
                                # 从subgraph_2中的对应位置提取patch，并交换到image_patch_1
                                max_patch_1 = image_patch_2[i][node_indices_2[z]]
                                image_patch_1[i][node_indices_1[j]] = max_patch_1

                # 遍历batch_indices_2中的每个元素，检查它是否属于当前批次
                for j in range(len(batch_indices_2)):
                    if batch_indices_2[j] == i:  # 当前批次
                        # 在batch_indices_1中查找与node_indices_2[j]对应的元素
                        for z in range(len(batch_indices_1)):
                            if batch_indices_1[z] == i and node_indices_1[z] == node_indices_2[j]:
                                # 从subgraph_1中的对应位置提取patch，并交换到image_patch_2
                                max_patch_2 = image_patch_1[i][node_indices_1[z]]
                                image_patch_2[i][node_indices_2[j]] = max_patch_2         

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([24, 224, 224])
    return image_patch_last

def Overlap(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, p, args):
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ',p1=args.patch_size, p2=args.patch_size)  # torch.Size([12, 224, 224])
    image_patch_2 = rearrange(net_input_2.squeeze(1),'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(out_1, 'b c (h p1) (w p2)->b c (h  w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(out_2, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])

    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)
            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])

            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
                kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

           # 获取 image_patch_2 的原始值
            max_patch_1 = image_patch_2[i][c_ori]  
            # 将 max_patch_1 和 image_patch_1[i][b] 混合
            image_patch_1[i][b] = (1-p) * max_patch_1 + p * image_patch_1[i][b]

            # 获取 image_patch_1 的原始值
            max_patch_2 = image_patch_1[i][a_ori]  
            # 将 max_patch_2 和 image_patch_2[i][d] 混合
            image_patch_2[i][d] = (1-p) * max_patch_2 + p * image_patch_2[i][d]
        else:
            a = torch.argmax(patches_mean_1[i].detach(), dim=0)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            c = torch.argmax(patches_mean_2[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)

            max_patch_1 = image_patch_2[i][c]  
            image_patch_1[i][b] = (1-p) * max_patch_1 + p * image_patch_1[i][b]  
            max_patch_2 = image_patch_1[i][a]
            image_patch_2[i][d] = (1-p) * max_patch_2 + p * image_patch_2[i][d]
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([24, 224, 224])
    return image_patch_last

def calculate_adjacent_confidence_differences_layer(patches_mean):
    if patches_mean.dim() == 4:  # If patches_mean has 4 dimensions (N, b, n, c)
        patches_mean = torch.mean(patches_mean, dim=3)  # Take mean across the channel dimension (c)

    N, b, n = patches_mean.shape  # N: number of samples, b: batch size, n: number of patches (h * w)
    h = w = int(n**0.5)  # Assuming patches are arranged in a square grid
    confidence_diff = torch.zeros_like(patches_mean)

    # For each patch, calculate the difference with its adjacent patches
    for i in range(h):
        for j in range(w):
            idx = i * w + j  # Current patch index
            neighbors = []
            if i > 0:  # Top neighbor
                neighbors.append(patches_mean[:, :, (i-1) * w + j])
            if i < h - 1:  # Bottom neighbor
                neighbors.append(patches_mean[:, :, (i+1) * w + j])
            if j > 0:  # Left neighbor
                neighbors.append(patches_mean[:, :, i * w + (j-1)])
            if j < w - 1:  # Right neighbor
                neighbors.append(patches_mean[:, :, i * w + (j+1)])
            if neighbors:
                # Compute the mean of the differences
                neighbor_diff = [torch.abs(patches_mean[:, :, idx] - neighbor) for neighbor in neighbors]
                confidence_diff[:, :, idx] = torch.mean(torch.stack(neighbor_diff, dim=0), dim=0)

    # Reshape the output back to the original shape (N, b, n, c)
    confidence_diff = confidence_diff.unsqueeze(-1).expand(-1, -1, -1, patches_mean.shape[-1])
    return confidence_diff

def weighted(label, p, p2):
    mask = torch.zeros_like(label)  # 创建与label相同形状的全零张量
    mask[label <= 0] = 1 - p             # 对大于0的元素赋值为1
    mask[label > 0] = 1 + p      # 对小于等于0的元素赋值为0.5
    # mask[label == 2] = 3*p 
    # mask[label == 3] = 4*p 
    return mask

def layer(outputs1_max, outputs2_max, outputs1_label, outputs2_label, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, p, p2, args, p_small):
    # ABD-I Bidirectional Displacement Patch
    # print(args.patch_size)
    patches_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    if p_small[p2] <= 64:
        patches_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64) ##修改的地方
    else:
        patches_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=128, p2=128)

    # patches2_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64)
    # patches2_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64) 
    # patches_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64)
    # patches_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64) ##修改的地方
    # image_1 = F.interpolate(volume_batch, scale_factor=0.0625, mode='bilinear', align_corners=False)
    # image_2 = F.interpolate(volume_batch_strong, scale_factor=0.0625, mode='bilinear', align_corners=False)
    # image_1 = rearrange(image_1.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
    # image_2 = rearrange(image_2.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
    label_patch_1 = rearrange(outputs1_label.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
    label_patch_2 = rearrange(outputs2_label.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
    
    image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
    B, N, _ = patches_1.size()
    B2, N2, _ = patches_2.size()
    # _, n, _ = patches2_1.size()

    # Rearrange patches to create smaller patches
    patches_small_1 = torch.stack([
        rearrange(patches_1[:, i, :], 
                  'b (p1s h_small p2s w_small) -> b (h_small w_small) (p1s p2s)', 
                  p1s=p_small[p2],
                  p2s=p_small[p2], 
                  h_small=args.patch_size // p_small[p2], 
                  w_small=args.patch_size // p_small[p2])
        for i in range(N)
    ], dim=0)

    patches_small_2 = torch.stack([
        rearrange(patches_2[:, i, :], 
                  'b (p1s h_small p2s w_small) -> b (h_small w_small) (p1s p2s)', 
                  p1s=p_small[p2],
                  p2s=p_small[p2], 
                  h_small=64 // p_small[p2], 
                  w_small=64 // p_small[p2])
        for i in range(N2)
    ], dim=0)###修改

    # patches_small2_1 = torch.stack([
    #     rearrange(patches2_1[:, i, :], 
    #               'b (p1s h_small p2s w_small) -> b (h_small w_small) (p1s p2s)', 
    #               p1s=p_small[p2],
    #               p2s=p_small[p2], 
    #               h_small=64 // p_small[p2], 
    #               w_small=64 // p_small[p2])
    #     for i in range(N)
    # ], dim=0)

    # patches_small2_2 = torch.stack([
    #     rearrange(patches2_2[:, i, :], 
    #               'b (p1s h_small p2s w_small) -> b (h_small w_small) (p1s p2s)', 
    #               p1s=p_small[p2],
    #               p2s=p_small[p2], 
    #               h_small=64 // p_small[p2], 
    #               w_small=64 // p_small[p2])
    #     for i in range(N)
    # ], dim=0)

    # print(patches_2.shape)
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # Original mean calculation
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)
    # print(patches_mean_2.shape)

    # patches_mean2_1 = torch.mean(patches2_1.detach(), dim=2)  # Original mean calculation
    # patches_mean2_2 = torch.mean(patches2_2.detach(), dim=2)

    # patches_small_mean_1 = torch.mean(patches_small2_1.detach(), dim=3)
    # patches_small_mean_2 = torch.mean(patches_small2_2.detach(), dim=3)

    patches_small_mean_1 = torch.mean(patches_small_1.detach(), dim=3)
    # print(patches_small_2.shape)
    patches_small_mean_2 = torch.mean(patches_small_2.detach(), dim=3)
    # print(patches_small_mean_2.shape)

    confidence_diff_1 = calculate_adjacent_confidence_differences_layer(patches_small_mean_1)
    # print(confidence_diff_1.shape)
    confidence_diff_2 = calculate_adjacent_confidence_differences_layer(patches_small_mean_2)

    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=64, p2=64)###修改

    # patches_outputs2_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=64, p2=64)
    # patches_outputs2_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=64, p2=64)

    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)

    # patches_mean_outputs2_1 = torch.mean(patches_outputs2_1.detach(), dim=3).permute(0, 2, 1)
    # patches_mean_outputs2_2 = torch.mean(patches_outputs2_2.detach(), dim=3).permute(0, 2, 1)

    # confidence_diff_outputs_1 = calculate_adjacent_confidence_differences(patches_mean_1)
    # confidence_diff_outputs_2 = calculate_adjacent_confidence_differences(patches_mean_2)

    # Use confidence_diff_1 and confidence_diff_2 for top-k sorting instead of original means
    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1, largest=True)
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1, largest=True)

    # patches_mean_1_top4_values, patches_mean_1_top42_indices = patches_mean2_1.topk(args.top_num, dim=1, largest=False)
    # patches_mean_2_top4_values, patches_mean_2_top42_indices = patches_mean2_2.topk(args.top_num, dim=1, largest=False)
    mask_cache = {}
    
    def create_circular_mask(patch_size):
        if patch_size not in mask_cache:
            radius = patch_size // 2
            y, x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))
            center = (radius, radius)
            dist = (x - center[0]).float()**2 + (y - center[1]).float()**2
            mask = (dist <= radius**2).float()
            mask_cache[patch_size] = mask
        return mask_cache[patch_size]

    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)

            kl_similarities2_1 = torch.empty(args.top_num)
            kl_similarities2_2 = torch.empty(args.top_num)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)
            # print(patches_mean_2.shape)

            # b2 = torch.argmin(patches_mean2_1[i].detach(), dim=0)
            # d2 = torch.argmin(patches_mean2_2[i].detach(), dim=0)

            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])

            # patches_mean_outputs_min2_1 = patches_mean_outputs2_1[i, b, :]  # torch.Size([4])
            # patches_mean_outputs_min2_2 = patches_mean_outputs2_2[i, d, :]  # torch.Size([4])

            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])
# )
#             patches_mean_outputs_top42_1 = patches_mean_outputs2_1[i, patches_mean_1_top42_indices[i, :], :]  # torch.Size([4, 4])
#             patches_mean_outputs_top42_2 = patches_mean_outputs2_2[i, patches_mean_2_top42_indices[i, :], :]  # torch.Size([4, 4]

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
                kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

                # kl_similarities2_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top42_1[j].softmax(dim=-1).log(), patches_mean_outputs_min2_2.softmax(dim=-1), reduction='sum')
                # kl_similarities2_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top42_2[j].softmax(dim=-1).log(), patches_mean_outputs_min2_1.softmax(dim=-1), reduction='sum')

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

            a2 = torch.argmin(kl_similarities2_1.detach(), dim=0, keepdim=False)
            c2 = torch.argmin(kl_similarities2_2.detach(), dim=0, keepdim=False)

            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

            # a_ori2 = patches_mean_1_top42_indices[i, a]
            # c_ori2 = patches_mean_2_top42_indices[i, c]

            b_small = torch.argmin(patches_small_mean_1[b][i].detach(), dim=0)
            d_small = torch.argmin(patches_small_mean_2[d][i].detach(), dim=0)
            a_small_ori = torch.argmax(patches_small_mean_1[a_ori][i].detach(), dim=0)
            c_small_ori = torch.argmax(patches_small_mean_2[c_ori][i].detach(), dim=0)



            # min_patch_1 = image_patch_2[i][d_small]  
            # # max_label_1 = label_patch_2[i][c_small_ori].to(device='cuda')
            # # weighted_1 = weighted(max_label_1,p,p2)
            # # max_patch_1 = max_patch_1 * weighted_1
            # image_patch_1[i][a_small_ori] = min_patch_1  
            # min_patch_2 = image_patch_1[i][b_small]
            # # max_label_2 = label_patch_1[i][a_small_ori].to(device='cuda')
            # # weighted_2 = weighted(max_label_2,p,p2)
            # # max_patch_2 = max_patch_2 * weighted_2
            # image_patch_2[i][c_small_ori] = min_patch_2 
            current_patch_size = p_small[p2]
            
            # 创建对应的圆形掩码并发送到GPU
            mask = create_circular_mask(current_patch_size).to(image_patch_1.device)
            mask = rearrange(mask, 'h w -> (h w)')
            
            # 修改置换操作（示例修改一组置换）
            # 原始代码：
            # image_patch_1[i][a_small_ori] = min_patch_1  
            # image_patch_2[i][c_small_ori] = min_patch_2
            
            # 修改后：
            # 源patch和目标patch
            source_patch = image_patch_2[i][d_small].clone()
            target_patch = image_patch_1[i][a_small_ori].clone()
            
            # 应用圆形掩码混合
            mixed_patch = target_patch * (1 - mask) + source_patch * mask
            image_patch_1[i][a_small_ori] = mixed_patch

            source_patch = image_patch_1[i][b_small].clone()
            target_patch = image_patch_2[i][c_small_ori].clone()
            
            mixed_patch = target_patch * (1 - mask) + source_patch * mask
            image_patch_2[i][c_small_ori] = mixed_patch
        else:
            # kl_similarities_1 = torch.empty(args.top_num)
            # kl_similarities_2 = torch.empty(args.top_num)
            # b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            # d = torch.argmin(patches_mean_1[i].detach(), dim=0)
            # patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
            # patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])

            # patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
            # patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

            # for j in range(args.top_num):
            #     kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
            #     kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

            # a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            # c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

            # a_ori = patches_mean_1_top4_indices[i, a]
            # c_ori = patches_mean_2_top4_indices[i, c]

            # b_small = torch.argmin(patches_small_mean_1[b][i].detach(), dim=0)
            # d_small = torch.argmin(patches_small_mean_2[d][i].detach(), dim=0)
            # a_small_ori = torch.argmax(patches_small_mean_1[a_ori][i].detach(), dim=0)
            # c_small_ori = torch.argmax(patches_small_mean_2[c_ori][i].detach(), dim=0)


            # max_patch_1 = image_2.squeeze(1)[i]
            # image_patch_1[i][b_small] = max_patch_1  
            # max_patch_2 = image_1.squeeze(1)[i]
            # image_patch_2[i][d_small] = max_patch_2 
            z1 = torch.argmin(patches_mean_1[i].detach(), dim=0)
            z2 = torch.argmin(patches_mean_2[i].detach(), dim=0)
            # print(patches_mean_2.shape)
            # print(patches_small_mean_2.shape)
            z3 = torch.argmax(patches_mean_1[i].detach(), dim=0)
            z4 = torch.argmax(patches_mean_2[i].detach(), dim=0)
            a = torch.argmin(patches_small_mean_1[z1][i].detach(), dim=0)
            b = torch.argmin(patches_small_mean_2[z2][i].detach(), dim=0)
            c = torch.argmax(patches_small_mean_1[z3][i].detach(), dim=0)
            d = torch.argmax(patches_small_mean_2[z4][i].detach(), dim=0)

            # # 将第二组中的最难块放入第一组中最易块的位置
            # max_patch_1 = image_patch_2[i][d]  
            # # max_label_1 = label_patch_2[i][d]
            # # weighted_1 = weighted(max_label_1,p,p2)
            # # max_patch_1 = max_patch_1 * weighted_1
            # image_patch_1[i][a] = max_patch_1  
            # max_patch_2 = image_patch_1[i][c]
            # # max_label_2 = label_patch_1[i][c]
            # # weighted_2 = weighted(max_label_2,p,p2)
            # # max_patch_2 = max_patch_2 * weighted_2
            # image_patch_2[i][b] = max_patch_2 
            current_patch_size = p_small[p2]
            mask = create_circular_mask(current_patch_size).to(image_patch_1.device)
            mask = rearrange(mask, 'h w -> (h w)')
            # 原始代码：
            # image_patch_1[i][a] = max_patch_1  
            # image_patch_2[i][b] = max_patch_2
            
            # 修改后：
            source_patch = image_patch_2[i][d].clone()
            target_patch = image_patch_1[i][a].clone()
            # print(target_patch.shape)
            # print(mask.shape)
            # print(source_patch.shape)
            mixed_patch = target_patch * (1 - mask) + source_patch * mask
            image_patch_1[i][a] = mixed_patch

            source_patch = image_patch_1[i][c].clone()
            target_patch = image_patch_2[i][b].clone()
            mixed_patch = target_patch * (1 - mask) + source_patch * mask
            image_patch_2[i][b] = mixed_patch

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h_small w_small)(p1s p2s) -> b  (h_small p1s) (w_small p2s)', h_small=args.image_size[0] // p_small[p2], w_small=args.image_size[0] // p_small[p2],p1s=p_small[p2], p2s=p_small[p2]) 
    return image_patch_last

# def layer_2(outputs1_max, outputs2_max, outputs1_label, outputs2_label, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, p, p2, args, p_small):
#     # ABD-I Bidirectional Displacement Patch
#     patches_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
#     patches_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64) ##修改的地方

#     # patches2_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64)
#     # patches2_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64) 
#     # patches_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64)
#     # patches_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=64, p2=64) ##修改的地方
#     # image_1 = F.interpolate(volume_batch, scale_factor=0.0625, mode='bilinear', align_corners=False)
#     # image_2 = F.interpolate(volume_batch_strong, scale_factor=0.0625, mode='bilinear', align_corners=False)
#     # image_1 = rearrange(image_1.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
#     # image_2 = rearrange(image_2.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
#     label_patch_1 = rearrange(outputs1_label.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
#     label_patch_2 = rearrange(outputs2_label.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])

#     # image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=16, p2s=16)
#     # image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])

#     image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
#     image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h_small p1s) (w_small p2s) -> b (h_small w_small) (p1s p2s) ', p1s=p_small[p2], p2s=p_small[p2])
#     B, N, _ = patches_1.size()
#     # _, n, _ = patches2_1.size()

#     # Rearrange patches to create smaller patches
#     patches_small_1 = torch.stack([
#         rearrange(patches_1[:, i, :], 
#                   'b (p1s h_small p2s w_small) -> b (h_small w_small) (p1s p2s)', 
#                   p1s=p_small[p2],
#                   p2s=p_small[p2], 
#                   h_small=args.patch_size // p_small[p2], 
#                   w_small=args.patch_size // p_small[p2])
#         for i in range(N)
#     ], dim=0)

#     patches_small_2 = torch.stack([
#         rearrange(patches_2[:, i, :], 
#                   'b (p1s h_small p2s w_small) -> b (h_small w_small) (p1s p2s)', 
#                   p1s=p_small[p2],
#                   p2s=p_small[p2], 
#                   h_small=64 // p_small[p2], 
#                   w_small=64 // p_small[p2])
#         for i in range(N)
#     ], dim=0)

#     # patches_small2_1 = torch.stack([
#     #     rearrange(patches2_1[:, i, :], 
#     #               'b (p1s h_small p2s w_small) -> b (h_small w_small) (p1s p2s)', 
#     #               p1s=p_small[p2],
#     #               p2s=p_small[p2], 
#     #               h_small=64 // p_small[p2], 
#     #               w_small=64 // p_small[p2])
#     #     for i in range(N)
#     # ], dim=0)

#     # patches_small2_2 = torch.stack([
#     #     rearrange(patches2_2[:, i, :], 
#     #               'b (p1s h_small p2s w_small) -> b (h_small w_small) (p1s p2s)', 
#     #               p1s=p_small[p2],
#     #               p2s=p_small[p2], 
#     #               h_small=64 // p_small[p2], 
#     #               w_small=64 // p_small[p2])
#     #     for i in range(N)
#     # ], dim=0)

#     patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # Original mean calculation
#     patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

#     # patches_mean2_1 = torch.mean(patches2_1.detach(), dim=2)  # Original mean calculation
#     # patches_mean2_2 = torch.mean(patches2_2.detach(), dim=2)

#     # patches_small_mean_1 = torch.mean(patches_small2_1.detach(), dim=3)
#     # patches_small_mean_2 = torch.mean(patches_small2_2.detach(), dim=3)

#     patches_small_mean_1 = torch.mean(patches_small_1.detach(), dim=3)
#     patches_small_mean_2 = torch.mean(patches_small_2.detach(), dim=3)

#     confidence_diff_1 = calculate_adjacent_confidence_differences_layer(patches_small_mean_1)
#     # print(confidence_diff_1.shape)
#     confidence_diff_2 = calculate_adjacent_confidence_differences_layer(patches_small_mean_2)

#     patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=p_small[p2], p2=p_small[p2])
#     patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=p_small[p2], p2=p_small[p2])

#     # patches_outputs2_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=64, p2=64)
#     # patches_outputs2_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=64, p2=64)

#     patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)
#     patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)

#     # patches_mean_outputs2_1 = torch.mean(patches_outputs2_1.detach(), dim=3).permute(0, 2, 1)
#     # patches_mean_outputs2_2 = torch.mean(patches_outputs2_2.detach(), dim=3).permute(0, 2, 1)

#     # confidence_diff_outputs_1 = calculate_adjacent_confidence_differences(patches_mean_1)
#     # confidence_diff_outputs_2 = calculate_adjacent_confidence_differences(patches_mean_2)

#     # Use confidence_diff_1 and confidence_diff_2 for top-k sorting instead of original means
#     # patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1, largest=False)
#     # patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1, largest=False)


#     # patches_mean_1_top4_values, patches_mean_1_top42_indices = patches_mean2_1.topk(args.top_num, dim=1, largest=False)
#     # patches_mean_2_top4_values, patches_mean_2_top42_indices = patches_mean2_2.topk(args.top_num, dim=1, largest=False)

#     for i in range(args.labeled_bs):
#         if random.random() < 0.5:
#             kl_similarities_1 = torch.empty(args.top_num)
#             kl_similarities_2 = torch.empty(args.top_num)

#             kl_similarities2_1 = torch.empty(args.top_num)
#             kl_similarities2_2 = torch.empty(args.top_num)
#             b = torch.argmin(patches_mean_1[i].detach(), dim=0)
#             d = torch.argmin(patches_mean_2[i].detach(), dim=0)
#             b_small = torch.argmin(patches_small_mean_1[b][i].detach(), dim=0)
#             d_small = torch.argmin(patches_small_mean_2[d][i].detach(), dim=0)

#             a = torch.argmax(patches_mean_1[i].detach(), dim=0)
#             c = torch.argmax(patches_mean_2[i].detach(), dim=0)

#             # b2 = torch.argmin(patches_mean2_1[i].detach(), dim=0)
#             # d2 = torch.argmin(patches_mean2_2[i].detach(), dim=0)
#             patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_small_mean_1[a].topk(args.top_num, dim=1, largest=False)
#             patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_small_mean_2[c].topk(args.top_num, dim=1, largest=False)

#             patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b_small, :]  # torch.Size([4])
#             patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d_small, :]  # torch.Size([4])

#             # patches_mean_outputs_min2_1 = patches_mean_outputs2_1[i, b, :]  # torch.Size([4])
#             # patches_mean_outputs_min2_2 = patches_mean_outputs2_2[i, d, :]  # torch.Size([4])

#             patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
#             patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])
# # )
# #             patches_mean_outputs_top42_1 = patches_mean_outputs2_1[i, patches_mean_1_top42_indices[i, :], :]  # torch.Size([4, 4])
# #             patches_mean_outputs_top42_2 = patches_mean_outputs2_2[i, patches_mean_2_top42_indices[i, :], :]  # torch.Size([4, 4]

#             for j in range(args.top_num):
#                 kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
#                 kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

#                 # kl_similarities2_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top42_1[j].softmax(dim=-1).log(), patches_mean_outputs_min2_2.softmax(dim=-1), reduction='sum')
#                 # kl_similarities2_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top42_2[j].softmax(dim=-1).log(), patches_mean_outputs_min2_1.softmax(dim=-1), reduction='sum')

#             a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
#             c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

#             # a2 = torch.argmin(kl_similarities2_1.detach(), dim=0, keepdim=False)
#             # c2 = torch.argmin(kl_similarities2_2.detach(), dim=0, keepdim=False)

#             a_ori = patches_mean_1_top4_indices[i, a]
#             c_ori = patches_mean_2_top4_indices[i, c]

#             # a_ori2 = patches_mean_1_top42_indices[i, a]
#             # c_ori2 = patches_mean_2_top42_indices[i, c]

            
#             # a_small_ori = torch.argmax(patches_small_mean_1[a_ori][i].detach(), dim=0)
#             # c_small_ori = torch.argmax(patches_small_mean_2[c_ori][i].detach(), dim=0)



#             max_patch_1 = image_patch_2[i][c_ori]  
#             # max_label_1 = label_patch_2[i][c_small_ori].to(device='cuda')
#             # weighted_1 = weighted(max_label_1,p,p2)
#             # max_patch_1 = max_patch_1 * weighted_1
#             image_patch_1[i][b_small] = max_patch_1  
#             max_patch_2 = image_patch_1[i][a_ori]
#             # max_label_2 = label_patch_1[i][a_small_ori].to(device='cuda')
#             # weighted_2 = weighted(max_label_2,p,p2)
#             # max_patch_2 = max_patch_2 * weighted_2
#             image_patch_2[i][d_small] = max_patch_2 
#         else:
#             # kl_similarities_1 = torch.empty(args.top_num)
#             # kl_similarities_2 = torch.empty(args.top_num)
#             # b = torch.argmin(patches_mean_1[i].detach(), dim=0)
#             # d = torch.argmin(patches_mean_1[i].detach(), dim=0)
#             # patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
#             # patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])

#             # patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
#             # patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

#             # for j in range(args.top_num):
#             #     kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
#             #     kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

#             # a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
#             # c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

#             # a_ori = patches_mean_1_top4_indices[i, a]
#             # c_ori = patches_mean_2_top4_indices[i, c]

#             # b_small = torch.argmin(patches_small_mean_1[b][i].detach(), dim=0)
#             # d_small = torch.argmin(patches_small_mean_2[d][i].detach(), dim=0)
#             # a_small_ori = torch.argmax(patches_small_mean_1[a_ori][i].detach(), dim=0)
#             # c_small_ori = torch.argmax(patches_small_mean_2[c_ori][i].detach(), dim=0)


#             # max_patch_1 = image_2.squeeze(1)[i]
#             # image_patch_1[i][b_small] = max_patch_1  
#             # max_patch_2 = image_1.squeeze(1)[i]
#             # image_patch_2[i][d_small] = max_patch_2 
#             z1 = torch.argmin(patches_mean_1[i].detach(), dim=0)
#             z2 = torch.argmin(patches_mean_2[i].detach(), dim=0)
#             z3 = torch.argmax(patches_mean_1[i].detach(), dim=0)
#             z4 = torch.argmax(patches_mean_2[i].detach(), dim=0)
#             a = torch.argmin(patches_small_mean_1[z1][i].detach(), dim=0)
#             b = torch.argmin(patches_small_mean_2[z2][i].detach(), dim=0)
#             c = torch.argmax(patches_small_mean_1[z3][i].detach(), dim=0)
#             d = torch.argmax(patches_small_mean_2[z4][i].detach(), dim=0)

#             # 将第二组中的最难块放入第一组中最易块的位置
#             max_patch_1 = image_patch_2[i][d]  
#             # max_label_1 = label_patch_2[i][d]
#             # weighted_1 = weighted(max_label_1,p,p2)
#             # max_patch_1 = max_patch_1 * weighted_1
#             image_patch_1[i][a] = max_patch_1  
#             max_patch_2 = image_patch_1[i][c]
#             # max_label_2 = label_patch_1[i][c]
#             # weighted_2 = weighted(max_label_2,p,p2)
#             # max_patch_2 = max_patch_2 * weighted_2
#             image_patch_2[i][b] = max_patch_2 

#     image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
#     image_patch_last = rearrange(image_patch, 'b (h_small w_small)(p1s p2s) -> b  (h_small p1s) (w_small p2s)', h_small=args.image_size[0] // p_small[p2], w_small=args.image_size[0] // p_small[p2],p1s=p_small[p2], p2s=p_small[p2]) 
#     return image_patch_last

def hhh(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args):

    patches_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) ##修改的地方
    image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # Original mean calculation
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # Calculate adjacent confidence differences
    # print(patches_mean_1.shape)
    confidence_diff_1 = calculate_adjacent_confidence_differences(patches_mean_1)
    confidence_diff_2 = calculate_adjacent_confidence_differences(patches_mean_2)
    # print(confidence_diff_1.shape)

    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)
    # confidence_diff_outputs_1 = calculate_adjacent_confidence_differences(patches_mean_outputs_1)
    # confidence_diff_outputs_2 = calculate_adjacent_confidence_differences(patches_mean_outputs_2)

    # Use confidence_diff_1 and confidence_diff_2 for top-k sorting instead of original means
    patches_mean_1_top4_values, patches_mean_1_top4_indices = confidence_diff_1.topk(args.top_num, dim=1, largest=False)
    patches_mean_2_top4_values, patches_mean_2_top4_indices = confidence_diff_2.topk(args.top_num, dim=1, largest=False)
    
    mask_cache = {}

    def create_circular_mask(patch_size):
        if patch_size not in mask_cache:
            radius = patch_size // 2
            y, x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))
            center = (radius, radius)
            dist = (x - center[0]).float()**2 + (y - center[1]).float()**2
            mask = (dist <= radius**2).float()
            mask_cache[patch_size] = mask
        return mask_cache[patch_size]
    
    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)
            # print("confidence_diff_2.shape:", confidence_diff_2.shape)
            b = torch.argmax(confidence_diff_1[i].detach(), dim=0)
            d = torch.argmax(confidence_diff_2[i].detach(), dim=0)
            # 修改这个部分的索引，使其与 confidence_diff_outputs_1 的维度一致
            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b]  # 移除第三个索引
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d]  # 移除第三个索引
            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :]]  # 无需第三个索引
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :]]  # 无需第三个索引

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
                kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)
            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

            current_patch_size = args.patch_size
            
            # 创建对应的圆形掩码并发送到GPU
            mask = create_circular_mask(current_patch_size).to(image_patch_1.device)
            mask = rearrange(mask, 'h w -> (h w)')
            
            # 修改置换操作（示例修改一组置换）
            # 原始代码：
            # image_patch_1[i][a_small_ori] = min_patch_1  
            # image_patch_2[i][c_small_ori] = min_patch_2
            
            # 修改后：
            # 源patch和目标patch
            source_patch = image_patch_2[i][d].clone()
            target_patch = image_patch_1[i][a_ori].clone()
            
            # 应用圆形掩码混合
            mixed_patch = target_patch * (1 - mask) + source_patch * mask
            image_patch_1[i][a_ori] = mixed_patch

            source_patch = image_patch_1[i][b].clone()
            target_patch = image_patch_2[i][c_ori].clone()
            
            mixed_patch = target_patch * (1 - mask) + source_patch * mask
            image_patch_2[i][c_ori] = mixed_patch

        else:
            hardest_patch_1 = torch.argmax(confidence_diff_1, dim=1)
            easiest_patch_1 = torch.argmin(confidence_diff_1, dim=1)
            hardest_patch_2 = torch.argmax(confidence_diff_2, dim=1)
            easiest_patch_2 = torch.argmin(confidence_diff_2, dim=1)

            current_patch_size = args.patch_size
            
            # 创建对应的圆形掩码并发送到GPU
            mask = create_circular_mask(current_patch_size).to(image_patch_1.device)
            mask = rearrange(mask, 'h w -> (h w)')
            
            # 修改置换操作（示例修改一组置换）
            # 原始代码：
            # image_patch_1[i][a_small_ori] = min_patch_1  
            # image_patch_2[i][c_small_ori] = min_patch_2
            
            # 修改后：
            # 源patch和目标patch
            source_patch = image_patch_2[i][hardest_patch_2].clone()
            target_patch = image_patch_1[i][easiest_patch_1].clone()
            
            # 应用圆形掩码混合
            mixed_patch = target_patch * (1 - mask) + source_patch * mask
            image_patch_1[i][easiest_patch_1] = mixed_patch

            source_patch = image_patch_1[i][hardest_patch_1].clone()
            target_patch = image_patch_2[i][easiest_patch_2].clone()
            
            mixed_patch = target_patch * (1 - mask) + source_patch * mask
            image_patch_2[i][easiest_patch_2] = mixed_patch 

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size) 
    return image_patch_last