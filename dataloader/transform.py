import cv2
import numpy as np
import albumentations as album
from albumentations.pytorch import ToTensorV2


def build_transformer(image_size=(416, 416)):
    transformers = {}
    transformers['train'] = album.Compose([
            album.RandomSizedBBoxSafeCrop(height=image_size[0], width=image_size[1]),
            album.ShiftScaleRotate(rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=(127,127,127), p=1.0),
            album.HorizontalFlip(p=0.5),
            album.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            album.RandomBrightnessContrast(p=0.3),
            album.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
            ToTensorV2()],
        bbox_params=album.BboxParams(format='yolo', min_area=25, min_visibility=0.05, label_fields=['class_ids']),
    )
    transformers['val'] = album.Compose([
            album.Resize(height=image_size[0], width=image_size[1]),
            album.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
            ToTensorV2()],
        bbox_params=album.BboxParams(format='yolo', label_fields=['class_ids']),
    )
    return transformers


def transform_square_image(image, bboxes, pad_const=127):
    pad_h, pad_w = 0, 0
    img_h, img_w, img_c = image.shape
    longest = max(img_h, img_w)
    
    if img_h < longest:
        pad_h = longest - img_h
    if img_w < longest:
        pad_w = longest - img_w

    square_image = np.ones(shape=(img_h+pad_h, img_w+pad_w, img_c), dtype=image.dtype) * pad_const
    square_image[:img_h, :img_w, :] = image
    bboxes[:, 0] *= (img_w / (img_w+pad_w))
    bboxes[:, 1] *= (img_h / (img_h+pad_h))
    return square_image, bboxes, (pad_h, pad_w)