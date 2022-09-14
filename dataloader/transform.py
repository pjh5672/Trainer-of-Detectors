import math
import random

import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms.functional as TF


IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation



class Albumentations:
    def __init__(self, p_flipud=0.0, p_fliplr=0.5):
        self.transform = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.VerticalFlip(p=p_flipud),
            A.HorizontalFlip(p=p_fliplr),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))

    def __call__(self, image, target):
        transform_data = self.transform(image=image, bboxes=target[:, 1:5], class_ids=target[:, 0])
        image = transform_data['image']
        bboxes = np.array(transform_data['bboxes'], dtype=np.float32)
        class_ids = np.array(transform_data['class_ids'], dtype=np.float32)
        target = np.concatenate((class_ids[:, np.newaxis], bboxes), axis=1)
        return image, target


def augment_hsv(image, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    dtype = image.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)



def random_perspective(image, target, input_size=416, degrees=0, translate=0.1, scale=0.1, perspective=0.0):
    image_h, image_w = image.shape[:2]
    max_side = max(image_h, image_w) if max(image_h, image_w) > input_size else input_size
    
    # Center
    C = np.eye(3)
    C[0, 2] = -image_w / 2  # x translation (pixels)
    C[1, 2] = -image_h / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * image_w  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * image_h  # y translation (pixels)

    # Combined rotation matrix
    M = T @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    image = cv2.warpPerspective(image, M, dsize=(max_side, max_side), borderValue=(0, 0, 0))

    n = len(target)
    xy = np.ones((n * 4, 3))
    xy[:, :2] = target[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ M.T  # transform
    xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # perspective rescale or affine

    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, max_side)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, max_side)

    idx = box_candidates(box1=target[:, 1:5].T * s, box2=new.T, wh_thr=4)
    target = target[idx]
    target[:, 1:5] = new[idx]
    return image, target


def random_crop(image, target, x_max, y_max, crop_size=416):
    x = random.randint(0, x_max - crop_size)
    y = random.randint(0, y_max - crop_size)
    image = image[y:y+crop_size, x:x+crop_size]
    target[:, [1,3]] -= x 
    target[:, [2,4]] -= y
    target[:, 1:5] = target[:, 1:5].clip(min=0, max=crop_size)
    return image, target


def mixup(image1, target1, image2, target2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    image = (image1 * r + image2 * (1 - r)).astype(np.uint8)
    target = np.concatenate((target1, target2), 0)
    return image, target


def box_candidates(box1, box2=None, wh_thr=4, ar_thr=20, area_thr=0.05, eps=1e-16):  # box1(4,n), box2(4,n)
    if box2 is not None:
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
    else:
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        ar = np.maximum(w1 / (h1 + eps), h1 / (w1 + eps))  # aspect ratio
        return (w1 > wh_thr) & (h1 > wh_thr) & (ar < ar_thr)  # candidates


def transform_square_image(image, bboxes):
    pad_h, pad_w = 0, 0
    img_h, img_w, img_c = image.shape
    max_size = max(img_h, img_w)

    if img_h < max_size:
        pad_h = max_size - img_h
    if img_w < max_size:
        pad_w = max_size - img_w

    square_image = np.zeros(shape=(img_h+pad_h, img_w+pad_w, img_c), dtype=image.dtype)
    square_image[:img_h, :img_w, :] = image
    bboxes[:, [0,2]] *= (img_w / (img_w + pad_w))
    bboxes[:, [1,3]] *= (img_h / (img_h + pad_h))
    return square_image, bboxes, max_size


def to_tensor(image):
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    image = torch.from_numpy(image).float()
    image /= 255.
    return image


def to_image(tensor):
    tensor *= 255.
    image = tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image


def normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    tensor = TF.normalize(image, mean, std)
    return tensor


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    return denorm_tensor
