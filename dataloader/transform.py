import cv2
import numpy as np
import albumentations as album
from albumentations.pytorch import ToTensorV2


def build_transformer(input_size=(416, 416)):
    input_h, input_w = input_size
    strongness = 0.2

    transformers = {}
    transformers['train'] = album.Compose([
        album.HorizontalFlip(p=0.5),
        album.RandomSizedCrop(min_max_height=[input_h-int(input_w*strongness),input_h+int(input_w*strongness)], height=input_h, width=input_w, w2h_ratio=1.0, p=1.0),
        album.ShiftScaleRotate(rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=1.0),
        album.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
        album.RandomBrightnessContrast(p=0.3),
        album.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2()],
        bbox_params=album.BboxParams(format='yolo', min_area=25, min_visibility=0.05, label_fields=['class_ids']),
    )
    transformers['val'] = album.Compose([
        album.HorizontalFlip(p=0.5),
        album.Resize(height=input_h, width=input_w),
        album.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2()],
        bbox_params=album.BboxParams(format='yolo', label_fields=['class_ids']),
    )
    return transformers


def transform_square_image(image, bboxes):
    pad_h, pad_w = 0, 0
    img_h, img_w, img_c = image.shape
    max_side = max(img_h, img_w)
    
    if img_h < max_side:
        pad_h = max_side - img_h
    if img_w < max_side:
        pad_w = max_side - img_w

    square_image = np.zeros(shape=(img_h+pad_h, img_w+pad_w, img_c), dtype=image.dtype)
    square_image[:img_h, :img_w, :] = image
    bboxes[:, [0,2]] *= (img_w / (img_w+pad_w))
    bboxes[:, [1,3]] *= (img_h / (img_h+pad_h))
    return square_image, bboxes, (max_side, pad_h, pad_w)