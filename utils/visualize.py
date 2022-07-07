import random

import cv2

TEXT_COLOR = (255, 255, 255)



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
    x_center, y_center, w, h = label[1:5]
    x_min = int(x_center - w/2)
    y_min = int(y_center - h/2)
    x_max = int(x_center + w/2)
    y_max = int(y_center + h/2)
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