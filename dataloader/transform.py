import math
import random

import cv2
import numpy as np
import albumentations as A
import torchvision.transforms.functional as TF

from utils import *

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation



class Transformer():
    def __init__(self, phase, input_size=416):
        self.phase = phase
        self.input_size = input_size
        self.h_gain=0.015
        self.s_gain=0.7
        self.v_gain=0.4
        self.degrees=15
        self.translate=0.15
        self.scale=0.9
        self.perspective=0.0001
        self.augmentor = self.build_albumentation() if self.phase == 'train' else None


    def __call__(self, image, target):
        max_side = 0
        if self.phase == 'train':
            image = self.augment_hsv(image, hgain=self.h_gain, sgain=self.s_gain, vgain=self.v_gain)
            transform_data = self.augmentor(image=image, bboxes=target[:, 1:5], class_ids=target[:, 0])
            image = transform_data['image']
            bboxes = np.array(transform_data['bboxes'], dtype=np.float32)
            class_ids = np.array(transform_data['class_ids'], dtype=np.float32)
            target = np.concatenate((class_ids[:, np.newaxis], bboxes), axis=1)
            target[:, 1:5] = box_transform_xcycwh_to_x1y1x2y2(target[:, 1:5])
            target[:, 1:5] = scale_to_original(target[:, 1:5], scale_w=image.shape[1], scale_h=image.shape[0])
            image, target = self.random_perspective(image, target, degrees=self.degrees, translate=self.translate, scale=self.scale, perspective=self.perspective)
            if len(target) == 0:
                target = np.array([[-1, 0., 0., image.shape[1], image.shape[0]]], dtype=np.float32)
            target[:, 1:5] = scale_to_norm(target[:, 1:5], image_w=image.shape[1], image_h=image.shape[0])
            target[:, 1:5] = box_transform_x1y1x2y2_to_xcycwh(target[:, 1:5])
        else:
            image, target[:, 1:5], max_side = self.transform_square_image(image, target[:, 1:5])
        image = cv2.resize(image, dsize=(self.input_size, self.input_size))
        tensor = normalize(to_tensor(image))
        return tensor, target, max_side


    def build_albumentation(self):
        augmentor = A.Compose([
            A.Blur(p=0.05),
            A.MedianBlur(p=0.05),
            A.ToGray(p=0.05),
            A.CLAHE(p=0.05),
            A.HorizontalFlip(p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))
        return augmentor


    def augment_hsv(self, image, hgain=0.5, sgain=0.5, vgain=0.5):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


    def random_perspective(self, image, target, degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0):
        image_h, image_w = image.shape[:2]
        max_side = max(image_h, image_w)
        
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

        idx = self.box_candidates(box1=target[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        target = target[idx]
        target[:, 1:5] = new[idx]
        return image, target


    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
    
    
    def transform_square_image(self, image, bboxes):
        pad_h, pad_w = 0, 0
        img_h, img_w, img_c = image.shape
        max_side = max(img_h, img_w)

        if img_h < max_side:
            pad_h = max_side - img_h
        if img_w < max_side:
            pad_w = max_side - img_w

        square_image = np.zeros(shape=(img_h+pad_h, img_w+pad_w, img_c), dtype=image.dtype)
        square_image[:img_h, :img_w, :] = image
        bboxes[:, [0,2]] *= (img_w / (img_w + pad_w))
        bboxes[:, [1,3]] *= (img_h / (img_h + pad_h))
        return square_image, bboxes, max_side


def to_tensor(image):
    image = np.ascontiguousarray(image.transpose(2, 0, 1)) # HWC -> CHW
    image = torch.from_numpy(image).float()  # to torch
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
