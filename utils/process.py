import numpy as np


def filter_obj_score(prediction, conf_threshold=0.01):
    valid_index = (prediction[:, 4] >= conf_threshold)
    bboxes = prediction[:, :4][valid_index]
    conf_scores = prediction[:, 4][valid_index]
    class_ids = np.argmax(prediction[:, 5:][valid_index], axis=1)
    return np.concatenate([class_ids[:, np.newaxis], bboxes, conf_scores[:, np.newaxis]], axis=-1)


def hard_NMS(bboxes, scores, iou_threshold):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    pick = []
    
    while len(order) > 0:
        pick.append(order[0])
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h)
        ious = overlap / (areas[order[0]] + areas[order[1:]] - overlap + 1e-8)
        order = order[np.where(ious < iou_threshold)[0] + 1]
    return pick


def run_NMS(prediction, iou_threshold, maxDets=100, class_agnostic=False):
    if len(prediction) == 0:
        return []

    if class_agnostic:
        pick = hard_NMS(prediction[:, 1:5], prediction[:, 5], iou_threshold)
        return prediction[pick[:maxDets]]

    prediction_multi_class = []
    for cls_id in np.unique(prediction[:, 0]):
        pred_per_cls_id = prediction[prediction[:, 0] == cls_id]
        pick_per_cls_id = hard_NMS(pred_per_cls_id[:, 1:5], pred_per_cls_id[:, 5], iou_threshold)
        prediction_multi_class.append(pred_per_cls_id[pick_per_cls_id])
    prediction_multi_class = np.concatenate(prediction_multi_class, axis=0)
    order = prediction_multi_class[:, -1].argsort()[::-1]
    return prediction_multi_class[order[:maxDets]]
