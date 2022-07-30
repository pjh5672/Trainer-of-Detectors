import numpy as np


def scale_to_original(bboxes, scale_w, scale_h):
    bboxes[:,[0,2]] *= scale_w
    bboxes[:,[1,3]] *= scale_h
    return bboxes.round(2)


def filter_obj_score(prediction, conf_threshold=0.01):
    valid_index = (prediction[:, 4] >= conf_threshold)
    bboxes = prediction[:, :4][valid_index]
    conf_scores = prediction[:, 4][valid_index]
    class_ids = np.argmax(prediction[:, 5:][valid_index], axis=1)
    return np.concatenate([class_ids[:, np.newaxis], bboxes, conf_scores[:, np.newaxis]], axis=-1)


def run_NMS_for_yolo(prediction, iou_threshold=0.1):
    bboxes = prediction[:, 1:5]
    scores = prediction[:, 5]

    if len(bboxes) == 0:
        return []
        
    if bboxes.dtype.kind == "i":
        bboxes = bboxes.astype("float")
        
    x1 = bboxes[:, 0] - bboxes[:, 2]/2
    y1 = bboxes[:, 1] - bboxes[:, 3]/2
    x2 = bboxes[:, 0] + bboxes[:, 2]/2
    y2 = bboxes[:, 1] + bboxes[:, 3]/2
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    pick = []
    while len(order) > 0:
        i = order[0]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = (w * h) + 1e-10
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        idxs = np.where(overlap <= iou_threshold)[0]
        order = order[idxs + 1]
    return prediction[pick]


