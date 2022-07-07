import numpy as np


def denormalize(input_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    tensor = input_tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor.clamp_(min=0, max=1.)
    tensor *= 255.
    image = tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image


def filter_obj_score(prediction, conf_threshold=0.01):
    valid_index = (prediction[..., 4] > conf_threshold)
    bboxes = prediction[:, :4][valid_index]
    conf_scores = prediction[:, 4][valid_index]
    class_ids = np.argmax(prediction[:, 5:][valid_index], axis=1)
    return np.concatenate([class_ids[:, np.newaxis], bboxes, conf_scores[:, np.newaxis]], axis=-1)


def run_NMS_for_yolo(prediction, iou_threshold=0.1):
    bboxes = prediction[:, 1:5]
    if len(bboxes) == 0:
        return []
    if bboxes.dtype.kind == "i":
        bboxes = bboxes.astype("float")
        
    pick = []
    x1 = bboxes[:, 0] - bboxes[:, 2]/2
    y1 = bboxes[:, 1] - bboxes[:, 3]/2
    x2 = bboxes[:, 0] + bboxes[:, 2]/2
    y2 = bboxes[:, 1] + bboxes[:, 3]/2
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0])))
    return prediction[pick]