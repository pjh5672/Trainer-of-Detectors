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


def get_IoU_target_with_anchor(wh1, wh2):
    w1, h1 = wh1
    w2, h2 = wh2
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area


def check_best_possible_recall(dataloader, PBR_params, anchor_iou_threshold=0.25):
    input_size, num_anchor_per_scale, anchors, strides = PBR_params
    total_n_target, total_n_train = 0, 0

    for index, mini_batch in enumerate(dataloader):
        targets = mini_batch[1]
        n_target_per_mini_batch, n_train_per_mini_batch = 0, 0

        for target in targets:
            target = torch.from_numpy(target)
            alined_anchor = np.zeros((len(target), 4))
            anchor_mask = np.zeros((len(target), len(anchors), 3))
            target_wh = target[:, 3:5] * input_size

            for scale_index in range(len(anchors)):
                anchor_each_scale = anchors[scale_index]
                grid_size = int(input_size / strides[scale_index])
                target_xy = target[:, 1:3] * grid_size
                iou_target_with_anchor = [get_IoU_target_with_anchor(target_wh.t(), anchor) for anchor in anchor_each_scale]
                iou_target_with_anchor = torch.stack(iou_target_with_anchor, dim=0)
                best_iou, _ = iou_target_with_anchor.max(dim=0)
                anchor_mask[:, scale_index, 0:2] = target_xy.long()
                anchor_mask[:, scale_index, 2] = best_iou

            anchor_index = np.argmax(anchor_mask[:,:,2], axis=1)
            alined_anchor[:, 2] = anchor_index
            alined_anchor[:, 3] = anchor_mask[np.arange(len(anchor_index)), anchor_index, 2]
            alined_anchor[:, 0:2] = anchor_mask[np.arange(len(anchor_index)), anchor_index, 0:2]
            assert len(alined_anchor) == len(target), 'Invalid anchors matching with targets'

            _, non_overlap_index = np.unique(alined_anchor[:, 0:3], axis=0, return_index=True)
            non_overlap_anchor = alined_anchor[non_overlap_index]
            best_possible_anchor = non_overlap_anchor[non_overlap_anchor[:, 3] > anchor_iou_threshold]
            n_train_per_mini_batch += len(best_possible_anchor)
            n_target_per_mini_batch += len(target)
            
        total_n_train += n_train_per_mini_batch
        total_n_target += n_target_per_mini_batch
        
    return total_n_train, total_n_target


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