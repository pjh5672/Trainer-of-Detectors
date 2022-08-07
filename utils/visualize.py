import random
import cv2
import numpy as np

from general import box_transform_xcycwh_to_x1y1x2y2, scale_to_original

TEXT_COLOR = (255, 255, 255)


def denormalize(input_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    tensor = input_tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor.clamp_(min=0, max=1.)
    tensor *= 255.
    image = tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image


def generate_random_color(num_colors):
    color_list = []
    for i in range(num_colors):
        hex_color = ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        rgb_color = tuple(int(hex_color[k:k+2], 16) for k in (0, 2, 4))
        color_list.append(rgb_color)
    return color_list


def visualize_bbox(image, label, class_list, color_list, 
                   show_class=False, show_score=False, fontscale=0.7, thickness=2):
    class_id = int(label[0])
    x_min, y_min, x_max, y_max = list(map(int, label[1:5]))
    color = color_list[class_id]
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    if show_class:
        class_name = class_list[class_id]
        if show_score:
            class_name += f'({label[-1]*100:.0f}%)'

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 2)
        cv2.rectangle(image, (x_min, y_min - int(fontscale*2 * text_height)), (x_min + text_width, y_min), color, -1)
        cv2.putText(image, text=class_name, org=(x_min, y_min - int((1-fontscale) * text_height)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, color=TEXT_COLOR, lineType=cv2.LINE_AA)
    return image
    

def visualize(image, label, class_list, color_list, show_class=False, show_score=False):
    canvas = image.copy()
    for item in label:
        canvas = visualize_bbox(canvas, item, class_list, color_list, show_class=show_class, show_score=show_score)
    return canvas


def visualize_prediction(tensor_image, prediction, conf_threshold, class_list, color_list):
    pred_yolo= prediction[1]
    input_size = tensor_image.shape[-1]
    canvas = denormalize(tensor_image)
    pred_voc = pred_yolo[pred_yolo[:, 5] >= conf_threshold].copy()
    pred_voc[:, 1:5] = box_transform_xcycwh_to_x1y1x2y2(pred_voc[:, 1:5])
    pred_voc[:, 1:5] = scale_to_original(pred_voc[:, 1:5], scale_w=input_size, scale_h=input_size)
    canvas = visualize(canvas, pred_voc, class_list, color_list, show_class=True, show_score=True)
    return canvas[...,::-1]