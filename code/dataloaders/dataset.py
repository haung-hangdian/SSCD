import os
# import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from dataloaders.augmentation import *

class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print("Train total {} samples".format(len(self.sample_list)))

        elif self.split == "val":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print("Val total {} samples".format(len(self.sample_list)))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class BaseDataSetsnii(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print("Train total {} samples".format(len(self.sample_list)))

        elif self.split == "val":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print("Val total {} samples".format(len(self.sample_list)))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        # with h5py.File(self._base_dir + "/mr_train_image/slice/{}".format(case), "r") as h5f_image:
        #     print("文件中的数据集:")
        #     print(list(h5f_image.keys()))  # 打印所有的键（数据集或组的名称）
        if self.split == "train":
            h5f_image = h5py.File(self._base_dir + "/mr_train_image/slice/{}".format(case), "r")
            h5f_label = h5py.File(self._base_dir + "/mr_train_label07/slice/{}".format(case), "r")
        else:
            h5f_image = h5py.File(self._base_dir + "/test_image/{}.h5".format(case), "r")
            h5f_label = h5py.File(self._base_dir + "/test_label/{}.h5".format(case), "r")
        image = h5f_image['data'][:]
        label = h5f_label['data'][:]
        image = image.astype(np.float32)
        label = label.astype(np.uint8)
        # print(label.max())
        # print(image.dtype)
        # print(label.dtype)
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample
   
class WnetAugmentation(object):
    def __init__(self, size=224, mean=(0.5),std=(0.3),\
                 scale=(0.64, 1),\
                 ratio_crop=(4. / 5., 5. / 4.),\
                 ratio_expand=(1,2),\
                 ratio_noise=30):
        self.augment = Compose([
                                ConvertFromInts(), ## 变成float
                                # Xiaobo(),
                                PhotometricDistort_grey(delta=ratio_noise,), ## delta control noise intensity
                                # # RandomElastic(9,0.08),  ##弹性增强
                                # Expand(0,ratio = ratio_expand), # expand ratio 加黑边
                                Rotate(360),
                                RandomResizedCrop(size=size,scale=scale,ratio=ratio_crop), ## scale
                                
                                Resize(size),
                                RandomMirror(),
                                
                                # RandomAffine(degrees=0, translate=None, scale=None ,shear=[-10,10,-10,10], fillcolor=(0, 0, 0)),
                                ToTensor(),
                                # Normalize(mean,std),
                            ])
        # self.augment_weakly = Compose([
        #                         ConvertFromInts(), ## 变成float
        #                         # Xiaobo(),
        #                         # PhotometricDistort_grey(delta=ratio_noise,), ## delta control noise intensity
        #                         # # RandomElastic(9,0.08),  ##弹性增强
        #                         # Expand(0,ratio = ratio_expand), # expand ratio 加黑边
        #                         Rotate(360),
        #                         # RandomResizedCrop(size=size,scale=scale,ratio=ratio_crop), ## scale
        #                         Resize(size),
        #                         RandomMirror(),
                                
        #                         # RandomAffine(degrees=0, translate=None, scale=None ,shear=[-10,10,-10,10], fillcolor=(0, 0, 0)),
        #                         ToTensor(),
        #                         # Normalize(mean,std),
        #                     ])

    def __call__(self, img, boxes, labels, masks):
        
        return self.augment(img, boxes, labels, masks)

def get_transform(self):
        return WnetAugmentation(size=self.opt.fineSize,
                                mean=self.opt.mean,std=self.opt.std,\
                                 scale=self.opt.scale,\
                                 ratio_crop=self.opt.ratio_crop,\
                                 ratio_expand=self.opt.ratio_expand,\
                                 ratio_noise=self.opt.ratio_noise)

class TeethBaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(os.path.join(self._base_dir, "train_slices.list"), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, "val.list"), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image_path = os.path.join(self._base_dir, "image", f"{case}.jpg")
            label_path = os.path.join(self._base_dir, "label", f"{case}.png")
        else:
            image_path = os.path.join(self._base_dir, "image", f"{case}.jpg")
            label_path = os.path.join(self._base_dir, "label", f"{case}.png")
        
        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")
        
        image = np.array(image)
        label = np.array(label)
        
        sample = {"image": image, "label": label}
        
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        
        sample["idx"] = idx
        return sample
                 
class TeethDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
            opt=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.opt = opt
        self.transform = self.get_transform()

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print("Train total {} samples".format(len(self.sample_list)))

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print("Val total {} samples".format(len(self.sample_list)))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        
    def get_transform(self):
        return WnetAugmentation(size=self.opt.fineSize,
                                mean=self.opt.mean,std=self.opt.std,\
                                 scale=self.opt.scale,\
                                 ratio_crop=self.opt.ratio_crop,\
                                 ratio_expand=self.opt.ratio_expand,\
                                 ratio_noise=self.opt.ratio_noise)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        # print(case)
        if self.split == "train":
            image_path = os.path.join(self._base_dir, "image", "{}.jpg".format(case))
            label_path = os.path.join(self._base_dir, "label", "{}.png".format(case))
        else:
            image_path = os.path.join(self._base_dir, "image", "{}.jpg".format(case))
            label_path = os.path.join(self._base_dir, "label", "{}.png".format(case))
        image = Image.open(image_path)
        label = Image.open(label_path)
        sample = {"image": image, "label": label}
        if None not in (self.ops_weak, self.ops_strong):
            sample = self.transform(sample, self.ops_weak, self.ops_strong)
        else:
            sample["image"],_,_,sample["label"] = self.transform(sample["image"],np.array([[0,0,1,1]]),np.array([[4]]),sample["label"])
                
        sample["idx"] = idx
        
        # print(f"Sample {idx} - image shape: {sample['image'].shape}, label shape: {sample['label'].shape}")
        return sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)  
    image = np.rot90(image, k)  
    axis = np.random.randint(0, 2)  
    image = np.flip(image, axis=axis).copy()  
    if label is not None:  
        label = np.rot90(label, k)  
        label = np.flip(label, axis=axis).copy()  
        return image, label
    else:
        return image

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)  
    image = ndimage.rotate(image, angle, order=0, reshape=False)  
    label = ndimage.rotate(label, angle, order=0, reshape=False)  
    return image, label

def blur(image,p=0.5):
    if random.random()<p:
        max = np.max(image)
        min = np.min(image)
        sigma = np.random.uniform(0.1,2.0)
        image = Image.fromarray(((image-min)/(max-min)*255).astype('uint8'))
        image = np.array(image.filter(ImageFilter.GaussianBlur(radius=sigma)))
        image = min + image*(max-min)/255
    return image
    
def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def cutout_gray(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=1, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.randint(value_min, value_max + 1, (erase_h, erase_w))
        else:
            value = np.random.randint(value_min, value_max + 1)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 0

    return img, mask

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:  
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # print(image.shape)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y),order=0)  
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class WeakStrongAugment(object):
    """returns weakly and strongly augmented images
    Args:
        object (tuple): output size of network
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:  
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # weak augmentation is rotation / flip
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y),order=0)  
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # strong augmentation is color jitter
        # image_strong2 = blur(image, p=0.5)
        image_strong, label_strong = cutout_gray(image,label,p=0.5)
        image_strong = color_jitter(image).type("torch.FloatTensor")
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        image_strong = image_strong.to(torch.float32)

        label = torch.from_numpy(label.astype(np.uint8))
        label_strong = torch.from_numpy(label_strong.astype(np.uint8))
        # label_strong = label
        sample = {
            "image": image,
            "image_strong": image_strong,
            "label": label,
            "label_strong": label_strong,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

# class WeakStrongAugment(object):
#     """returns weakly and strongly augmented images
#     Args:
#         object (tuple): output size of network
#     """

#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample["image"], sample["label"]
#         image = self.resize(image)
#         label = self.resize(label)
#         # weak augmentation is rotation / flip
#         image_weak, label = random_rot_flip(image, label)
#         # strong augmentation is color jitter
#         image_strong = color_jitter(image_weak).type("torch.FloatTensor")
#         # fix dimensions
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.uint8))

#         sample = {
#             "image": image,
#             "image_weak": image_weak,
#             "image_strong": image_strong,
#             "label_aug": label,
#         }
#         return sample

#     def resize(self, image):
#         x, y = image.shape
#         return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class WeakStrongAugmentTeeth(object):
    """Returns weakly and strongly augmented images.
    Args:
        output_size (tuple): Desired output size (height, width) for the images and labels.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Convert the image to grayscale (single channel)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Random rotation/flip as weak augmentation
        if random.random() > 0.5:  
            image_gray, label = random_rot_flip(image_gray, label)
        elif random.random() > 0.5:
            image_gray, label = random_rotate(image_gray, label)

        # Resize image and label to the desired output size
        h, w = label.shape
        image_gray = zoom(image_gray, (self.output_size[0] / h, self.output_size[1] / w), order=1)  # Use order=1 for bilinear interpolation
        label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)  # Use order=0 for nearest-neighbor interpolation

        # Strong augmentation: cutout and color jitter
        image_strong, label_strong = cutout_gray(image_gray, label, p=0.5)
        image_strong = color_jitter(image_strong).float()

        # Convert the augmented single-channel images back to three channels
        image_gray_3ch = np.stack([image_gray]*3, axis=-1)
        image_strong_3ch = np.stack([image_strong]*3, axis=-1)

        # Convert image and label to PyTorch tensors
        image_strong_3ch = torch.from_numpy(image_strong_3ch)
        if image_strong_3ch.dim() == 4:  # Check if the tensor has 4 dimensions
            image_strong_3ch = image_strong_3ch.squeeze(0)  # Remove the batch dimension
        image_gray_3ch = torch.from_numpy(image_gray_3ch.astype(np.float32)).permute(2, 0, 1)  # Change to (C, H, W)
        image_strong_3ch = image_strong_3ch.permute(2, 0, 1)

        label = torch.from_numpy(label.astype(np.uint8))
        label_strong = torch.from_numpy(label_strong.astype(np.uint8))

        # Return the augmented sample
        sample = {
            "image": image_gray_3ch,
            "image_strong": image_strong_3ch,
            "label": label,
            "label_strong": label_strong
        }
        return sample

    def resize(self, image):
        h, w = image.shape[:2]
        return zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=1)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size),
        )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class ACDCDataset(Dataset):
    """ ACDC Dataset
        input: base_dir -> your parent level path
               split -> "sup", "unsup" and "eval", must specified
    """

    def __init__(self, base_dir, data_dir,
                 split, num=None, config=None):
        self.data_dir = data_dir
        self._base_dir = base_dir
        self.sample_list = []
        if split == 'train':
            with open(self._base_dir + '/data/train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + '/data/test.list', 'r') as f:
                self.image_list = f.readlines()
        # https://github.com/LiheYoung/UniMatch/blob/583e32492b0ac150e0946b65864d2dcc642220b8/more-scenarios/medical/dataset/acdc.py#L32
        # we follow UniMatch to perform validation for both val & test to find the best checkpoint.
        # the final results are reported based on test set only.
        elif split == "eval":
            with open(self._base_dir + '/data/eval_test.list', 'r') as f:
                self.image_list = f.readlines()
        else:
            raise NotImplementedError

        self.image_list = [item.strip() for item in self.image_list][:-1]
        if num is not None:
            self.image_list = self.image_list[:num]
        self.aug = config.augmentation if split == 'train' else False
        self.training_transform = transforms.Compose([
            Normalise(),
            RandomColorJitter(dict(
                brightness=.5, contrast=.5,
                sharpness=.25,   color=.25
            )),
            RandomGenerator((256, 256)),
            RandomCrop((256, 256), (256, 256)),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self.data_dir + "/" + image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image = torch.from_numpy(image).float().numpy()
        label = torch.from_numpy(np.array(label)).long().numpy()
        sample = {'image': image, 'label': label}
        if not self.aug:
            return sample['image'], sample['label']
        return self.training_transform(sample)