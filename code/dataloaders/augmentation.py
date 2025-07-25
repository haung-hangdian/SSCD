from __future__ import division
import torch
from torchvision import transforms
import cv2
import numpy as np
import types
import torch
import torch.nn as nn
from numpy import random
from torchvision.transforms import functional as F
from PIL import Image, ImageFile
import math
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
from PIL import ImageEnhance, ImageFilter, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, masks=None):
        for t in self.transforms:
            img, boxes, labels, masks = t(img, boxes, labels, masks)
            # print(img.shape, masks.shape)
        return img, boxes, labels,masks


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None, masks=None):
        return self.lambd(img, boxes, labels,masks)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, masks=None):
        if type(image)==np.ndarray:
            return image.astype(np.float32), boxes, labels, masks
        else:
            return np.array(image,dtype=np.float32), boxes, labels, np.array(masks,dtype=np.float32)


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, masks=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels, masks


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None, masks=None):
        if len(image.shape)==3:
            height, width, channels = image.shape
        else:
            height, width= image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels,masks


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None, masks=None):
        if len(image.shape)==3:
            height, width, channels = image.shape
        else:
            height, width= image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels,masks


class Rotate(object):
    def __init__(self, angle = 360):
        self.angle = angle

    def __call__(self, image, boxes=None, labels=None, masks=None):
        
        rotate_angle = random.uniform(0,self.angle)
        
        height, width, = image.shape
        
        rot_mat = cv2.getRotationMatrix2D((height//2 ,width//2), rotate_angle, 1)
        
        image = cv2.warpAffine(image, rot_mat, (height,width))
        masks = cv2.warpAffine(masks, rot_mat, (height,width))
        # print(image.shape)

        
        #masks = np.array(masks>0,dtype=np.int)
        return image, boxes, labels, masks
    
class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None, masks=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        masks = cv2.resize(masks, (self.size,
                                 self.size))
        
        #masks = np.array(masks>0,dtype=np.int)
        return image, boxes, labels, masks



class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            image[:, :, 0] *= random.uniform(self.lower, self.upper)
            image[:, :, 2] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels,masks



class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, masks


class RandomAddNoise(object):
    def __init__(self,delta=30.0):
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            image += random.uniform(-self.delta, self.delta)
        return image, boxes, labels,masks
    
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels,masks


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None,masks=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels,masks


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels,masks


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels,masks



class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None, masks=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels,masks.cpu().numpy().astype(np.int)


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None,masks=None):
        # print(cvimage.shape)
        # print(masks.shape)
        # if len(cvimage.shape)==3:
        #     if len(masks.shape) == 3:
        #         if cvimage.max()>10:
        #             return torch.from_numpy(cvimage.copy().astype(np.float32))[:,:,:3].permute(2, 0, 1)/255.0, torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),torch.from_numpy(masks.copy()).permute(2, 0, 1)
        #         else:
        #             return torch.from_numpy(cvimage.copy().astype(np.float32))[:,:,:3].permute(2, 0, 1), torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),torch.from_numpy(masks.copy()).permute(2, 0, 1)
        #     else:
        #         if cvimage.max()>10:
        #             return torch.from_numpy(cvimage.copy().astype(np.float32))[:,:,:3].permute(2, 0, 1)/255.0, torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),torch.from_numpy(masks.copy())
        #         else:
        #             return torch.from_numpy(cvimage.copy().astype(np.float32))[:,:,:3].permute(2, 0, 1), torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),torch.from_numpy(masks.copy())

        if len(cvimage.shape)==3:
            if cvimage.max()>10:
                return torch.from_numpy((cvimage.copy() / 255.0).astype(np.float32))[:,:,:3].permute(2, 0, 1), torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),(torch.from_numpy(masks.copy()) / 255).long()
            else:
                return torch.from_numpy(cvimage.copy().astype(np.float32))[:,:,:3].permute(2, 0, 1), torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),(torch.from_numpy(masks.copy())).long()
        else:
            #print(cvimage.shape,cvimage.min())
            if cvimage.max()>10:
                return (torch.from_numpy(cvimage.copy())[:,:,]/255.0).unsqueeze(0), torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),torch.from_numpy(masks.copy() // 255).float()
                
            else:
                return torch.from_numpy(cvimage.copy())[:,:,].unsqueeze(0), torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),(torch.from_numpy(masks.copy()) // 255).float()
            
        
class To255(object):
    def __call__(self, cvimage, boxes=None, labels=None,masks=None):
        if cvimage.max()<10:
            return cvimage*255.0,boxes,labels,masks
        else:
            return cvimage,boxes,labels,masks
        
        
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self,sample_options = None):
        if sample_options ==None:
            self.sample_options = (
                # using entire original input image
                None,
                # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
                (0.1, None),
                (0.3, None),
                (0.7, None),
                (0.9, None),
                # randomly sample a patch
                (None, None),)
        else:
            self.sample_optioins = sample_options
    def __call__(self, image, boxes=None, labels=None,masks=None):
        if len(image.shape)==3:
            height, width, channels = image.shape
        else:
            height, width= image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                current_masks = masks
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]
                current_masks = current_masks[rect[1]:rect[3], rect[0]:rect[2]]                                              

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels, current_masks

            
class RandomResizedCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, size, scale=(0.04, 0.25), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        """ Change this according to the original image sie """
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w < img.shape[0] and h < img.shape[1]:
                i = random.randint(0, img.shape[1] - h)
                j = random.randint(0, img.shape[0] - w)
                return i, j, h, w

        # Fallback

        w = min(img.shape[0], img.shape[1])
        i = (img.shape[1] - w) // 2
        j = (img.shape[0] - w) // 2
        return i, j, w, w
        
    def __call__(self, image, boxes=None, labels=None,masks=None):
        while True:
            # max trails (50)
            for _ in range(50):
                current_image = image
                current_masks = masks
                left,top,w,h = self.get_params(current_image,self.scale,self.ratio)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                
                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                #overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                #if overlap.min() < min_iou and max_iou < overlap.max():
                    #continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2]]
                current_masks = current_masks[rect[1]:rect[3], rect[0]:rect[2]]      

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                #if not mask.any():
                    #continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels,current_masks

class Xiaobo(object):
    def __init__(self):
        self.conv_H = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_L = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.high_pass_filter = torch.tensor([[-1, -1, -1],
                                              [-1, 4, -1],
                                              [-1, -1, -1]]).float().unsqueeze(0).unsqueeze(0)
        self.gaussian_filter = torch.tensor([[1, 2, 1],
                                             [2, 4, 2],
                                             [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0) / 16
        self.conv_H.weight.data = self.high_pass_filter
        self.conv_L.weight.data = self.gaussian_filter
    def __call__(self, image, boxes=None, labels = None, masks = None):
        
        # if isinstance(image, np.ndarray):
        #     B = image
        # else:
        #     # 如果B_path是一个文件路径
        #     B = Image.open(image)
        # img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # print(image.shape)
        img = image[:, :, 2:3]
        img = np.repeat(img, 3, axis=2)
        image = image.transpose(2, 0, 1)
        image = image[0:1, :, :]
        # print(image.shape)
        # print(img.shape)
        img_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        filtered_img_tensor_H = self.conv_H(img_tensor)
        filtered_img_tensor_L = self.conv_L(img_tensor)
        filtered_img_tensor = filtered_img_tensor_H - filtered_img_tensor_L
        filtered_img = filtered_img_tensor.squeeze(0).squeeze(0).detach().numpy()
        filtered_img = (filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min())
        mean_value = filtered_img.max()
        thresholded_img = np.where(filtered_img < mean_value*0.2, 0, 1)
        y_coords, x_coords = np.where(thresholded_img == 0)
        r = 40
        num_circles = 1
        if len(x_coords) < num_circles:
            num_circles = len(x_coords)
        selected_indices = np.random.choice(len(x_coords), num_circles, replace=False)
        for index in selected_indices:
            cv2.circle(img, (x_coords[index], y_coords[index]), r, (0, 0, 0), -1)
        image = img[:, :, 2:3]
        image = np.repeat(image, 4, axis=2)
        # print(image.shape)
        return image, boxes, labels,masks


class Expand(object):
    def __init__(self, mean,ratio = (1,1.1)):
        self.mean = mean
        self.ratio = ratio

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if image.max() >10:
            mean = np.array(self.mean)*255.0
        if random.randint(2):
            return image, boxes, labels,masks
        
        if len(image.shape)==3:
            height, width, depth= image.shape
        else:
            height, width= image.shape
            depth = 0
        
        if self.ratio == None:
            ratio = random.uniform(1, 2)
        else:
            ratio = random.uniform(self.ratio[0],self.ratio[1])
            
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        
        if depth==0:
            expand_image = np.zeros((int(height*ratio), int(width*ratio)),dtype=image.dtype)
#             expand_image[:, :] = mean
        else:
            expand_image = np.zeros((int(height*ratio), int(width*ratio), depth),dtype=image.dtype)
#             expand_image[:, :, :] = mean
        
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image
        

        # print(masks.shape)
        if len(masks.shape) == 3 :
            masks = masks[:, :, 0]
        
        expand_masks = np.zeros((int(height*ratio), int(width*ratio)), dtype=masks.dtype)
        expand_masks[int(top):int(top + height),
                        int(left):int(left + width)] = masks

        masks = expand_masks

        #boxes = boxes.copy()
        #boxes[:, :2] += (int(left), int(top))
        #boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels, masks


class RandomMirror(object):
    def __call__(self, image, boxes=None, labels =None,masks=None):
        if len(image.shape)==3:
            _, width, _ = image.shape
        else:
            _, width = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            masks = masks[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels,masks


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self,  image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self,delta=18.0,con_lower=0.5, con_upper=1.5,sat_lower=0.5, sat_upper=1.5):
        self.pd = [
            RandomContrast(lower=con_lower, upper=con_upper),
            ConvertColor(transform='HSV'),
            RandomSaturation(lower=sat_lower, upper=sat_upper),
            RandomHue(delta=delta),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast(lower=con_lower, upper=con_upper)
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()


    def __call__(self, image, boxes, labels,masks):
        im = image.copy()
        im, boxes, labels,masks = self.rand_brightness(im, boxes, labels,masks)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1])
        im, boxes, labels,masks= distort(im, boxes, labels, masks)
        return self.rand_light_noise(im, boxes, labels, masks)

class PhotometricDistort_grey(object):
    def __init__(self,delta=18.0,con_lower=0.5, con_upper=1.5):
        self.rand_contrast = RandomContrast(lower=con_lower, upper=con_upper)
        self.rand_brightness = RandomBrightness()
        self.rand_noise = RandomAddNoise(delta)


    def __call__(self, image, boxes, labels,masks):
        im = image.copy()
        im, boxes, labels,masks = self.rand_brightness(im, boxes, labels,masks)
        
        #distort = Compose(self.pd[0])
        im, boxes, labels,masks= self.rand_contrast(im, boxes, labels, masks)
        im, boxes, labels,masks= self.rand_noise(im, boxes, labels, masks)
        #return self.rand_light_noise(im, boxes, labels, masks)
        return im, boxes, labels, masks



class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, boxes=None, labels=None ,masks = None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        # print(image.type)
        # print(masks.type)
        #print(image.shape,len(masks.shape))
        if image.shape[0]==1:
            #print(self.mean)
            return (image-self.mean)/self.std,boxes,labels,masks

        if len(masks.shape)==2:
            return F.normalize(image, self.mean, self.std),boxes,labels, masks
        
        return F.normalize(image, self.mean, self.std),boxes,labels,F.normalize(masks, self.mean, self.std)


# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = torch.tensor(mean).float().unsqueeze(-1).unsqueeze(-1)
#         self.std = torch.tensor(std).float().unsqueeze(-1).unsqueeze(-1)

#     def __call__(self, image, boxes=None, labels=None, masks=None):
#         image = image.float()
#         # print(image.shape)
#         masks = masks.permute(2, 0, 1)
        
#         if masks is not None and len(masks.shape) != 2:
#             masks = masks.float()
        
#         if image.shape[0] == 1:
#             return (image - self.mean) / self.std, boxes, labels, masks
#         elif masks is not None and len(masks.shape) == 2:
#             return F.normalize(image, self.mean.squeeze(), self.std.squeeze()), boxes, labels, masks
#         else:
#             return F.normalize(image, self.mean.squeeze(), self.std.squeeze()), boxes, labels, F.normalize(masks, self.mean.squeeze(), self.std.squeeze())
    
    
class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """
    # degree: rotate for the image; \in [-180, 180]; 旋转
    # translate: translation for the image, \in [0,1] 平移
    # scale: scale the image with center invariant, better \in (0,2] 放缩
    # shear: shear the image with dx or dy, w\in [-180, 180] 扭曲
    # eg.
    # preprocess1 = myTransforms.RandomAffine(degrees=0, translate=[0, 0.2], scale=[0.8, 1.2],
    #                                        shear=[-10, 10, -10, 10], fillcolor=(228, 218, 218))
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        
        if degrees < 0:
            raise ValueError("If degrees is a single number, it must be positive.")
        self.degrees = (-degrees, degrees)


        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

#     def __call__(self, img):
    def __call__(self, image, boxes=None, labels=None ,masks = None):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, image.size)
        return F.affine(image, *ret, resample=self.resample, fillcolor=self.fillcolor),boxes,labels,F.affine(mask, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)

class RandomElastic(object):
    """Random Elastic transformation by CV2 method on image by alpha, sigma parameter.
        # you can refer to:  https://blog.csdn.net/qq_27261889/article/details/80720359
        # https://blog.csdn.net/maliang_1993/article/details/82020596
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates
    Args:
        alpha (float): alpha value for Elastic transformation, factor
        if alpha is 0, output is original whatever the sigma;
        if alpha is 1, output only depends on sigma parameter;
        if alpha < 1 or > 1, it zoom in or out the sigma's Relevant dx, dy.
        sigma (float): sigma value for Elastic transformation, should be \ in (0.05,0.1)
        mask (PIL Image) if not assign, set None.
    """
    def __init__(self, alpha, sigma):
    
        assert 0.05 <= sigma <= 0.1, \
            "In pathological image, sigma should be in (0.05,0.1)"
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def RandomElasticCV2(img, alpha, sigma,boxes=None, labels=None , mask=None):
        alpha = img.shape[1] * alpha
        sigma = img.shape[1] * sigma
        if mask is not None:
            mask = np.array(mask).astype(np.float)
            #print(img.shape,mask.shape)
            img = np.concatenate((img, np.expand_dims(mask,2)), axis=2)

        shape = img.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        img = map_coordinates(img, indices, order=0, mode='reflect').reshape(shape)
        if mask is not None:
            return img[..., :3], boxes, labels,img[..., 3]
        else:
            return img

    def __call__(self, img, boxes=None, labels=None ,mask=None):
        return self.RandomElasticCV2(np.array(img), self.alpha, self.sigma, boxes, labels , mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)
        format_string += ', sigma={0}'.format(self.sigma)
        format_string += ')'
        return format_string
    
transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness,   color=ImageEnhance.Color
)

class Normalise(object):
    def __call__(self, sample):
        image = sample['image']
        return{'image': (image - image.min()) / (image.max()-image.min()), 'label': sample['label']}
    
class RandomColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, sample):
        img = sample['image']
        out = Image.fromarray((img * 255).astype(np.uint8))
        rand_num = np.random.uniform(0, 1, len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)
        out = np.array(out).astype(float) / 255.0
        sample.update({'image': out})
        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, resolution_size):
        self.output_size = output_size
        self.resolution_size = resolution_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # the data in ACDC has lower resolution and needed to be padding for all of them!
        # we set random here to perform the padding or skipping for TraCo.
        if random.random() < 0.15:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

            w, h = image.shape
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])

            cons_start_x = np.random.randint(0, w1) if w1 != 0 else w1
            cons_start_y = np.random.randint(0, h1) if h1 != 0 else h1

            # no-overlap issues
            cons_start_x = cons_start_x + int(w1/2) if w1 - cons_start_x > 256 else cons_start_x
            cons_start_y = cons_start_y + int(h1/2) if h1 - cons_start_y > 256 else cons_start_y
            cons_image = image[cons_start_x:cons_start_x + self.output_size[0],
                               cons_start_y:cons_start_y + self.output_size[1]]

            cons_label = label[cons_start_x:cons_start_x + self.output_size[0],
                               cons_start_y:cons_start_y + self.output_size[1]]

            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

            assert cons_image.shape == image.shape, print(cons_image.shape, image.shape)
            assert cons_label.shape == label.shape, print(cons_label.shape, label.shape)
            a = image[0 if cons_start_x < w1 else cons_start_x-w1:self.output_size[0]-(w1-cons_start_x) if cons_start_x < w1 else self.output_size[0],
                0 if cons_start_y < h1 else cons_start_y-h1:self.output_size[1]-(h1-cons_start_y) if cons_start_y < h1 else self.output_size[1]]

            b = cons_image[0 if cons_start_x > w1 else w1-cons_start_x:self.output_size[0]-(cons_start_x-w1) if cons_start_x > w1 else self.output_size[0],
                0 if cons_start_y > h1 else h1-cons_start_y:self.output_size[1]-(cons_start_y-h1) if cons_start_y > h1 else self.output_size[1]]
            assert np.all(np.equal(a, b)), "?"
        else:
            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            x, y = image.shape
            cons_image = image.copy()
            cons_label = label.copy()
            w1, h1 = x, y
            cons_start_x = w1
            cons_start_y = h1

        return {'image': image, 'label': label, 'cons_image': cons_image, 'cons_label': cons_label,
                'normal_range_x': [0 if cons_start_x < w1 else cons_start_x-w1, self.resolution_size[0]-(w1-cons_start_x) if cons_start_x < w1 else self.resolution_size[0]],
                'cons_range_x': [0 if cons_start_x > w1 else w1-cons_start_x, self.resolution_size[1]-(cons_start_x-w1) if cons_start_x > w1 else self.resolution_size[1]],
                'normal_range_y': [0 if cons_start_y < h1 else cons_start_y-h1, self.resolution_size[0]-(h1-cons_start_y) if cons_start_y < h1 else self.resolution_size[0]],
                'cons_range_y': [0 if cons_start_y > h1 else h1-cons_start_y, self.resolution_size[1]-(cons_start_y-h1) if cons_start_y > h1 else self.resolution_size[1],]}

