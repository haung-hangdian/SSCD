import argparse
import logging
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
import torch.optim as optim
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.config import get_config
from networks.net_factory import net_factory, BCP_net, BCP_net_changedropout
from networks.vision_transformer import SwinAgentUnet as ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC_layer_weight1_2', help='experiment_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model_name')
parser.add_argument('--model_2', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
parser.add_argument('--image_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--cfg', type=str,
                    default="/home/v1-4080s/hhy/ABD/code/configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+', )
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
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

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model_1 == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main,_ = net(input)
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)
    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)
    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)
    return first_metric, second_metric, third_metric, prediction

import numpy as np

def calculate_confidence_difference(confidence_map, predictions, labels):
    """
    计算每个像素的置信度差值，然后根据预测结果将正确和错误的点分别加入数组 a 和 b.
    
    参数:
    confidence_map: 置信度图像，形状为 (b, h, w)。
    predictions: 预测结果，形状为 (b, h, w)。
    labels: 真实标签，形状为 (b, h, w)。
    
    返回:
    avg_confidence_diff_correct: 每个batch中正确预测点的平均置信度差值。
    avg_confidence_diff_incorrect: 每个batch中错误预测点的平均置信度差值。
    correct_points_count: 每个batch中正确预测点的数量。
    incorrect_points_count: 每个batch中错误预测点的数量。
    """
    # 获取 batch 大小和图像的高度和宽度
    batch_size, h, w = confidence_map.shape
    
    # 初始化全局结果变量
    total_correct_conf_diff = 0.0
    total_incorrect_conf_diff = 0.0
    total_correct_points = 0
    total_incorrect_points = 0
    
    # 对每个 batch 进行处理
    for b in range(batch_size):
        a = []  # 存储正确预测点的置信度差值
        b_ = []  # 存储错误预测点的置信度差值
        
        # 遍历每个像素
        for i in range(1, h-1):  # 从1到h-1，避免边界问题
            for j in range(1, w-1):
                # 计算当前点的置信度差值，基于四周邻居
                neighbors = [
                    confidence_map[b, i-1, j], confidence_map[b, i+1, j], 
                    confidence_map[b, i, j-1], confidence_map[b, i, j+1]
                ]
                confidence_diff = np.abs(confidence_map[b, i, j] - np.mean(neighbors))

                # 与真实标签进行比较，决定加入哪个数组
                if predictions[b, i, j] == labels[b, i, j]:  # 预测正确
                    a.append(confidence_diff)
                else:  # 预测错误
                    b_.append(confidence_diff)
        
        # 计算当前batch的正确和错误预测点的置信度差值
        avg_conf_diff_correct = np.mean(a) if len(a) > 0 else 0.0
        avg_conf_diff_incorrect = np.mean(b_) if len(b_) > 0 else 0.0
        correct_points_count = len(a)
        incorrect_points_count = len(b_)
        
        # 累加每个 batch 的置信度差值乘以点数
        total_correct_conf_diff += avg_conf_diff_correct * correct_points_count
        total_incorrect_conf_diff += avg_conf_diff_incorrect * incorrect_points_count
        total_correct_points += correct_points_count
        total_incorrect_points += incorrect_points_count
    
    # 计算总体平均置信度差值
    avg_confidence_diff_correct = total_correct_conf_diff / total_correct_points if total_correct_points > 0 else 0.0
    avg_confidence_diff_incorrect = total_incorrect_conf_diff / total_incorrect_points if total_incorrect_points > 0 else 0.0
    
    return avg_confidence_diff_correct, avg_confidence_diff_incorrect, total_correct_points, total_incorrect_points

def test_single_volume_with_confidence(case, net, test_save_path, FLAGS):
    """
    对单个体积进行测试，并计算正确点和错误点的置信度差值。
    
    参数:
    case: 当前测试的病例编号。
    net: 训练好的网络模型。
    test_save_path: 保存预测结果的路径。
    FLAGS: 配置参数。
    
    返回:
    first_metric: 第一类的预测指标（如：Dice, IOU等）。
    second_metric: 第二类的预测指标。
    third_metric: 第三类的预测指标。
    avg_correct_conf_diff: 正确预测点的平均置信度差值。
    avg_incorrect_conf_diff: 错误预测点的平均置信度差值。
    correct_points_count: 正确预测点的数量。
    incorrect_points_count: 错误预测点的数量。
    """
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    # print(label.shape)
    prediction = np.zeros_like(label)
    confidence_map = np.zeros_like(label, dtype=np.float32)  # 初始化置信度图
    
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model_1 == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            
            # 置信度图：softmax结果的最大值作为置信度
            softmax_out = torch.softmax(out_main, dim=1)
            # print(softmax_out.shape)
            # print(softmax_out.min())
            max_confidence = torch.max(softmax_out, dim=1)[0].squeeze(0).cpu().detach().numpy()
            # print(max_confidence.min())
            # print(max_confidence.max())
            
            min_value = max_confidence.min()

            # Step 2: 执行缩放操作
            zoomed_confidence = zoom(max_confidence, (x / 256, y / 256), order=0)

            # Step 3: 替换缩放后图像中的 0 值，将其替换为缩放前图像的最小值
            # 如果图像中的 0 值是因为插值引入的，可以将 0 替换为 min_value
            zoomed_confidence[zoomed_confidence == 0] = min_value

            # 也可以选择替换所有小于原先最小值的点为 min_value
            zoomed_confidence[zoomed_confidence < min_value] = min_value
            confidence_map[ind] = zoomed_confidence

            
            # 预测结果
            out = torch.argmax(softmax_out, dim=1).squeeze(0).cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    
    # 计算不同类别的评价指标
    if np.sum(prediction == 1) == 0:
        first_metric = 0, 0, 0, 0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)
    if np.sum(prediction == 2) == 0:
        second_metric = 0, 0, 0, 0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)
    if np.sum(prediction == 3) == 0:
        third_metric = 0, 0, 0, 0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)
    
    # 计算正确点和错误点的平均置信度差值
    avg_correct_conf_diff, avg_incorrect_conf_diff, correct_points_count, incorrect_points_count = calculate_confidence_difference(confidence_map, prediction, label)

    return first_metric, second_metric, third_metric, avg_correct_conf_diff, avg_incorrect_conf_diff, correct_points_count, incorrect_points_count
def Inference_model1_with_confidence(FLAGS):
    print("——Starting the Model1 Prediction with Confidence——")
    
    # 读取图片列表
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    
    # 定义路径
    snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_train".format(FLAGS.exp, FLAGS.labeled_num)
    test_save_path = "/home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_train/{}_predictions_model1/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_1)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    
    # 加载模型
    net = BCP_net(in_chns=1, class_num=args.num_classes)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(FLAGS.model_1))
    # net.load_state_dict(torch.load(save_mode_path))
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    # 初始化总指标
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0

    # 初始化置信度差值和点数统计
    total_correct_conf_diff = 0.0
    total_incorrect_conf_diff = 0.0
    total_correct_points = 0
    total_incorrect_points = 0

    # 遍历图片列表
    for case in tqdm(image_list):
        # 计算每个图片的指标和置信度差值
        first_metric, second_metric, third_metric, avg_correct_conf_diff, avg_incorrect_conf_diff, correct_points_count, incorrect_points_count = test_single_volume_with_confidence(case, net, test_save_path, FLAGS)
        
        # 累加每个类别的指标
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        
        # 累加置信度差值和点数
        total_correct_conf_diff += avg_correct_conf_diff
        total_incorrect_conf_diff += avg_incorrect_conf_diff
        total_correct_points += correct_points_count
        total_incorrect_points += incorrect_points_count

    # 计算平均指标
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    average = (avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3
    print(avg_metric)
    print(average)

    # 计算整体平均置信度差值
    avg_correct_conf_diff_total = total_correct_conf_diff / total_correct_points if total_correct_points > 0 else 0.0
    avg_incorrect_conf_diff_total = total_incorrect_conf_diff / total_incorrect_points if total_incorrect_points > 0 else 0.0
    
    print(f"Average Correct Confidence Difference: {avg_correct_conf_diff_total}")
    print(f"Average Incorrect Confidence Difference: {avg_incorrect_conf_diff_total}")
    
    # 保存结果到文件
    with open(os.path.join(test_save_path, 'performance_with_confidence.txt'), 'w') as file:
        file.write(f"Average Metrics: {avg_metric}\n")
        file.write(f"Average Dice/IOU: {average}\n")
        file.write(f"Average Correct Confidence Difference: {avg_correct_conf_diff_total}\n")
        file.write(f"Average Incorrect Confidence Difference: {avg_incorrect_conf_diff_total}\n")
    
    return avg_metric, avg_correct_conf_diff_total, avg_incorrect_conf_diff_total

# def Inference_model1(FLAGS):
#     print("——Starting the Model1 Prediction——")
#     with open(FLAGS.root_path + '/test.list', 'r') as f:image_list = f.readlines()
#     image_list = sorted([item.replace('\n', '').split(".")[0]for item in image_list])
#     snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_train".format(FLAGS.exp, FLAGS.labeled_num)
#     test_save_path = "//home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_train/{}_predictions_model1/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_1)
#     if not os.path.exists(test_save_path):
#         os.makedirs(test_save_path)
#     net = BCP_net(in_chns=1, class_num=args.num_classes)
#     save_mode_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(FLAGS.model_1))
#     net.load_state_dict(torch.load(save_mode_path))
#     print("init weight from {}".format(save_mode_path))
#     net.eval()

#     first_total = 0.0
#     second_total = 0.0
#     third_total = 0.0
#     for case in tqdm(image_list):
#         first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
#         first_total += np.asarray(first_metric)
#         second_total += np.asarray(second_metric)
#         third_total += np.asarray(third_metric)
#     avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
#     average = (avg_metric[0]+avg_metric[1]+avg_metric[2])/3
#     print(avg_metric)
#     print(average)
#     with open(os.path.join(test_save_path, 'performance.txt'), 'w') as file:
#         file.write(str(avg_metric) + '\n')
#         file.write(str(average) + '\n')
#     return avg_metric
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

def Inference_model1(FLAGS):
    print("——Starting the Model1 Prediction——")
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_last_train".format(FLAGS.exp, FLAGS.labeled_num)
    test_save_path = "/home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_last_train/{}_predictions_model/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_1)

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    net = BCP_net_changedropout(in_chns=1, class_num=FLAGS.num_classes)
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(FLAGS.model_1))
    # load_net_opt(net, optimizer, save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    metrics_and_cases = []  # 用于存储指标和对应 case 名

    for case in tqdm(image_list):
        # Run inference and get metrics
        first_metric, second_metric, third_metric, prediction = test_single_volume(case, net, test_save_path, FLAGS)
        
        # Save metrics and case name
        avg_case_metric = (first_metric[0] + second_metric[0] + third_metric[0]) / 3
        metrics_and_cases.append((avg_case_metric, case))

        # Accumulate metrics
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)

        # Save prediction results
        prediction_save_path = os.path.join(test_save_path, f'{case}_prediction.nii.gz')
        save_nifti(prediction, prediction_save_path)

    # Compute average metrics
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    average = (avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3
    print(avg_metric)
    print(average)

    # Sort cases by metric and get the top 10
    metrics_and_cases.sort(key=lambda x: x[0], reverse=True)  # 按平均指标从高到低排序
    top_10_cases = [case for _, case in metrics_and_cases[:10]]

    # Save performance metrics
    with open(os.path.join(test_save_path, 'performance.txt'), 'w') as file:
        file.write(str(avg_metric) + '\n')
        file.write(str(average) + '\n')
        file.write("Top 10 cases by metric:\n")
        file.writelines([f"{case}\n" for case in top_10_cases])

    return avg_metric

def save_nifti(data, save_path):
    """
    Save prediction data as a NIfTI file.
    """
    import nibabel as nib
    nifti_image = nib.Nifti1Image(data, np.eye(4))  # Assuming identity affine, adjust if necessary
    nib.save(nifti_image, save_path)

def Inference_model2(FLAGS):
    print("——Starting the Model2 Prediction——")
    with open(FLAGS.root_path + '/test.list', 'r') as f:image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]for item in image_list])
    snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_train".format(FLAGS.exp, FLAGS.labeled_num)
    test_save_path = "/home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_train/{}_predictions_model2/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_2)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    # net = ViT_seg(config, img_size=FLAGS.image_size, num_classes=FLAGS.num_classes).cuda()
    net = BCP_net(in_chns=1, class_num=args.num_classes)
    # net.load_from(config)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(FLAGS.model_2))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric, pre = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    average = (avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3
    print(avg_metric)
    print(average)
    with open(os.path.join(test_save_path, 'performance.txt'), 'w') as file:
        file.write(str(avg_metric) + '\n')
        file.write(str(average) + '\n')
    return avg_metric

# def Inference_model2(FLAGS):
#     print("——Starting the Model2 Prediction——")
#     with open(FLAGS.root_path + '/test.list', 'r') as f:
#         image_list = f.readlines()
#     image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
#     snapshot_path = "/home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_train".format(FLAGS.exp, FLAGS.labeled_num)
#     test_save_path = "/home/v1-4080s/hhy/ABD/model/BCP/{}_{}_labeled/self_train/{}_predictions_model2/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_2)

#     if not os.path.exists(test_save_path):
#         os.makedirs(test_save_path)

#     net = BCP_net(in_chns=1, class_num=FLAGS.num_classes)
#     save_mode_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(FLAGS.model_2))
#     net.load_state_dict(torch.load(save_mode_path))
#     print("init weight from {}".format(save_mode_path))
#     net.eval()

#     first_total = 0.0
#     second_total = 0.0
#     third_total = 0.0
#     metrics_and_cases = []  # 用于存储指标和对应 case 名

#     for case in tqdm(image_list):
#         # Run inference and get metrics
#         first_metric, second_metric, third_metric, prediction = test_single_volume(case, net, test_save_path, FLAGS)
        
#         # Save metrics and case name
#         avg_case_metric = (first_metric[0] + second_metric[0] + third_metric[0]) / 3
#         metrics_and_cases.append((avg_case_metric, case))

#         # Accumulate metrics
#         first_total += np.asarray(first_metric)
#         second_total += np.asarray(second_metric)
#         third_total += np.asarray(third_metric)

#         # Save prediction results
#         prediction_save_path = os.path.join(test_save_path, f'{case}_prediction.nii.gz')
#         save_nifti(prediction, prediction_save_path)

#     # Compute average metrics
#     avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
#     average = (avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3
#     print(avg_metric)
#     print(average)

#     # Sort cases by metric and get the top 10
#     metrics_and_cases.sort(key=lambda x: x[0], reverse=True)  # 按平均指标从高到低排序
#     top_10_cases = [case for _, case in metrics_and_cases[:10]]

#     # Save performance metrics
#     with open(os.path.join(test_save_path, 'performance.txt'), 'w') as file:
#         file.write(str(avg_metric) + '\n')
#         file.write(str(average) + '\n')
#         file.write("Top 10 cases by metric:\n")
#         file.writelines([f"{case}\n" for case in top_10_cases])

#     return avg_metric

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    FLAGS = parser.parse_args()
    metric_model1 = Inference_model1(FLAGS)
    # metric_model2 = Inference_model2(FLAGS)
    # a, b, c = Inference_model1_with_confidence(FLAGS)