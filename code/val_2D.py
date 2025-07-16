import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.config import get_config
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torchvision.transforms as transforms
from networks.unet import Sam_unet

from PIL import Image,ImageDraw,ImageEnhance,ImageColor

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/teeth_origin_1280', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='teeth/teeth_1280', help='experiment_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model_name')
parser.add_argument('--model_2', type=str,
                    default='helper', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=140,
                    help='labeled data')
parser.add_argument('--image_size', type=list, default=[256,256],
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

def calculate_metric_percase_promise(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0

def calculate_metric_percase(pred, gt):
    pred[pred > 0 ] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.dc(pred, gt)
        hd95 = metric.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, float('inf') 
    
def test_single_volume_promise(image, label, net, classes):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input)[0], dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_promise(prediction == i, label == i))
    return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() ###MMWHS
    # image = np.transpose(image, (2, 0, 1))
    # label = np.transpose(label, (2, 0, 1))
    # print(label.dtype)
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        # print(ige.shape)
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        # print(x)
        # print(y)
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out= net(input)
            # print(out.shape)
            out= torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            # print(pred.dtype)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        # print((prediction == i).max())
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_result(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() ###MMWHS
    # image = np.transpose(image, (2, 0, 1))
    # label = np.transpose(label, (2, 0, 1))
    # print(label.dtype)
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        # print(ige.shape)
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        # print(x)
        # print(y)
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out, _= net(input)
            # print(out.shape)
            out= torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            # print(pred.dtype)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        # print((prediction == i).max())
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_sam(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() ###MMWHS
    # image = np.transpose(image, (2, 0, 1))
    # label = np.transpose(label, (2, 0, 1))
    # print(label.dtype)
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        # print(ige.shape)
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        # print(x)
        # print(y)
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            _, out= net(input)
            # print(out.shape)
            out= torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            # print(pred.dtype)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        # print((prediction == i).max())
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_teeth(image, label, net, classes, patch_size=[224, 224]):
    label = label.squeeze(0).cpu().detach().numpy()
    input = image.cuda().float()
    x, y = image.shape[1], image.shape[2]
    net.eval()
    with torch.no_grad():
        out = net(input)
        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
    metric_list = []
    # print(pred.shape)
    for i in range(0, classes):
        metric_list.append(calculate_metric_percase(
            out == i, label == i))
    return metric_list

def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_teeth(images, labels, net, classes, patch_size=[640, 640]):
    labels = labels.cpu().detach().numpy()
    inputs = images.type(torch.float32).cuda()
    net.eval()
    
    with torch.no_grad():
        outputs = net(inputs)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        outputs = outputs.cpu().detach().numpy()

    batch_size = labels.shape[0]
    metric_list = []

    for b in range(batch_size):
        label = labels[b]
        output = outputs[b]
        for i in range(classes):
            if np.any(label == i):  # 检查标签中是否存在当前类别
                metric_list.append(calculate_metric_percase(
            output == i, label == i))
            else:
                # 如果类别不存在，加入默认值 (0, float('inf'))
                metric_list.append((0, float('inf')))
    return metric_list

def Inference_model1(FLAGS):###用作推理的
    print("——Starting the Model1 Prediction——")
    with open(FLAGS.root_path + '/val.list', 'r') as f:image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]for item in image_list])
    snapshot_path = "/home/v1-4080s/hhy/ABD/model/{}_{}_labeled/unet".format(FLAGS.exp, FLAGS.labeled_num)
    test_save_path = "/home/v1-4080s/hhy/ABD/model/{}_{}_labeled/{}_predictions_model/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_1)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model_1, in_chns=3,class_num=FLAGS.num_classes)
    # net = Sam_unet(in_chans=3, num_class=2).cuda()
    # net = ViT_seg(config, img_size=FLAGS.image_size, num_classes=FLAGS.num_classes).cuda()
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model_1))
    # save_mode_path = os.path.join(snapshot_path, 'iter_153000.pth'.format(FLAGS.model_1))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    
    for case in tqdm(image_list):
        # print(image.shape)
        input = Image.open(os.path.join(FLAGS.root_path, "image", "{}.png".format(case)))###更改后缀名
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量，并将像素值归一化到 [0, 1]
        ])
        input = transform(input).unsqueeze(0).cuda()
        x, y = input.shape[1], input.shape[2]
        net.eval()
        with torch.no_grad():
            out = net(input)
            # print(out.shape)
            out = torch.softmax(out, dim=1).squeeze(0)
            # print(out.shape)
            out = out.cpu().detach().numpy()
            # print(out)
            # i = (out[1].max()-out[1].min()) // 2
            # out[1][out[1] > i] = 1
            # for i in range(len(out)):
            #     mean = (out[i].max()-out[i].min()) // 2
            #     out[i][out[i] > ] = 1
            #     out[i] = out[i] * 255
            #     Image.fromarray(out[i]).convert('L').save(test_save_path + f"{case}_{i}.png")
                
        class_map = np.argmax(out,axis=0)
        class_map = np.asarray(class_map, dtype = np.uint8)
        # print(class_map)
      
        class_map[class_map > 0] = 255
        class_map[class_map==0] = 0
        

  
        # class_map[class_map==0] = 0
        # class_map[class_map==1] = 1 + 50
        # class_map[class_map==2] = 2 + 50
        # class_map[class_map==3] = 3 + 50
        # class_map[class_map==4] = 4 + 50
        # class_map[class_map==5] = 5 + 50
        # class_map[class_map==6] = 6 + 50
        # class_map[class_map==7] = 7 + 50
        # class_map[class_map==8] = 8 + 50
        # class_map[class_map==9] = 9 + 50
        # class_map[class_map==10] = 10 + 50
        # class_map[class_map==11] = 11 + 50
        # class_map[class_map==12] = 12 + 50
        # class_map[class_map==13] = 13 + 50
        # class_map[class_map==14] = 14 + 50
        # class_map[class_map==15] = 15 + 50
        # class_map[class_map==16] = 16 + 50
        # class_map[class_map==17] = 17 + 50
        # class_map[class_map==18] = 18 + 50
        # class_map[class_map==19] = 19 + 50
        # class_map[class_map==20] = 20 + 50
        # class_map[class_map==21] = 21 + 50
        # class_map[class_map==22] = 22 + 50
        # class_map[class_map==23] = 23 + 50
        # class_map[class_map==24] = 24 + 50
        # class_map[class_map==25] = 25 + 50
        # class_map[class_map==26] = 26 + 50
        # class_map[class_map==27] = 27 + 50
        # class_map[class_map==28] = 28 + 50
        # class_map[class_map==29] = 29 + 50
        # class_map[class_map==30] = 30 + 50
        # class_map[class_map==31] = 31 + 50
        # class_map[class_map==32] = 32 + 50
        # class_map[class_map==33] = 33 + 50
        # class_map[class_map==34] = 34 + 50
        # class_map[class_map==35] = 35 + 50
        # class_map[class_map==36] = 36 + 50
        # class_map[class_map==37] = 37 + 50
        # class_map[class_map==38] = 38 + 50
        # class_map[class_map==39] = 39 + 50
        # class_map[class_map==40] = 40 + 50
        # class_map[class_map==41] = 41 + 50
        # class_map[class_map==42] = 42 + 50
        # class_map[class_map==43] = 43 + 50
        # class_map[class_map==44] = 44 + 50
        # class_map[class_map==45] = 45 + 50
        # class_map[class_map==46] = 46 + 50
        # class_map[class_map==47] = 47 + 50
        # class_map[class_map==48] = 48 + 50
        # class_map[class_map==49] = 49 + 50
        # class_map[class_map==50] = 50 + 50
        # class_map[class_map==51] = 51 + 50
        # class_map[class_map==52] = 52 + 50
        
        # 保存图片到目标目录
        Image.fromarray(class_map).convert('L').save(test_save_path + f"{case}.png")

def Inference(FLAGS):
    print("——Starting the Model Prediction——")

    input_folder = os.path.join("/home/v1-4080s/hhy/ABD/data/STS24_Train_Validation_00002")  # 输入图像文件夹路径
    test_save_path = "/home/v1-4080s/hhy/ABD/data/STS24_Train_Validation_00002/predictions_model/"
    snapshot_path = "/home/v1-4080s/hhy/ABD/model/{}_{}/unet".format(FLAGS.exp, FLAGS.labeled_num)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    # 初始化网络
    net = net_factory(net_type=FLAGS.model_1, in_chns=3, class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model_1))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    # 遍历文件夹中的所有 .jpg 文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    
    for image_file in tqdm(image_files):
        case = os.path.splitext(image_file)[0]  # 获取文件名（不含扩展名）
        input_image_path = os.path.join(input_folder, image_file)

        input = Image.open(input_image_path)  # 打开图像文件
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量，并将像素值归一化到 [0, 1]
        ])
        input = transform(input).unsqueeze(0).cuda()

        with torch.no_grad():
            out = net(input)
            out = torch.softmax(out, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

        # 获取类别预测的最大概率值索引
        class_map = np.argmax(out, axis=0)
        class_map = np.asarray(class_map, dtype=np.uint8)

        # 将非背景类的像素值设为255
        class_map[class_map > 0] = 255
        class_map[class_map == 0] = 0

        # 保存处理后的图像
        output_image_path = os.path.join(test_save_path, f"{case}.png")
        Image.fromarray(class_map).convert('L').save(output_image_path)
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    FLAGS = parser.parse_args()
    metric_model1 = Inference_model1(FLAGS)