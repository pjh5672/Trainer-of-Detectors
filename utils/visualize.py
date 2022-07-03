import cv2

BOX_COLOR = (250, 50, 50)
TEXT_COLOR = (255, 255, 255)

def visualize_bbox(image, bbox, class_name, color=BOX_COLOR, fontscale=0.7, thickness=2):
    x_center, y_center, w, h = bbox
    x_min = int(x_center - w/2)
    y_min = int(y_center - h/2)
    x_max = int(x_center + w/2)
    y_max = int(y_center + h/2)
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 2)
    
    cv2.rectangle(image, 
                  (x_min, y_min - int(fontscale*2 * text_height)), 
                  (x_min + text_width, y_min), 
                  color, -1)
    
    cv2.putText(image,
                text=class_name,
                org=(x_min, y_min - int((1-fontscale) * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontscale, 
                color=TEXT_COLOR, 
                lineType=cv2.LINE_AA)
    return image


def visualize(image, bboxes, class_ids, classname_list):
    canvas = image.copy()
    for bbox, class_id in zip(bboxes, class_ids):
        if not isinstance(class_id, int):
            class_id = int(class_id)
        class_name = classname_list[class_id]
        canvas = visualize_bbox(canvas, bbox, class_name)
    return canvas