import os
import cv2
import torch
import numpy as np



def save_model(model, save_path):
    torch.save(model, save_path)


def scale_to_original(bboxes, scale_w, scale_h):
    bboxes[:,[0,2]] *= scale_w
    bboxes[:,[1,3]] *= scale_h
    return bboxes.round(2)


def scale_to_norm(bboxes, image_w, image_h):
    bboxes[:,[0,2]] /= image_w
    bboxes[:,[1,3]] /= image_h
    return bboxes


def clip_box_coordinates(bboxes):
    bboxes = box_transform_xcycwh_to_x1y1x2y2(bboxes)
    bboxes = box_transform_x1y1x2y2_to_xcycwh(bboxes)
    return bboxes


def box_transform_xcycwh_to_x1y1x2y2(bboxes, clip_max=None):
    x1y1 = bboxes[:, :2] - bboxes[:, 2:] / 2
    x2y2 = bboxes[:, :2] + bboxes[:, 2:] / 2
    x1y1x2y2 = np.concatenate((x1y1, x2y2), axis=1)
    x1y1x2y2 = x1y1x2y2.clip(min=0., max=clip_max if clip_max is not None else 1.)
    return x1y1x2y2


def box_transform_x1y1x2y2_to_xcycwh(bboxes):
    wh = bboxes[:, 2:] - bboxes[:, :2]
    xcyc = bboxes[:, :2] + wh / 2
    xcycwh = np.concatenate((xcyc, wh), axis=1)
    return xcycwh


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False