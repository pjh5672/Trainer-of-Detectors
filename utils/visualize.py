import random
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import warnings
warnings.simplefilter("ignore", UserWarning)
plt.rcParams.update({'figure.max_open_warning': 0})

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
    pred_yolo = prediction[1]
    input_size = tensor_image.shape[-1]
    canvas = denormalize(tensor_image)
    pred_voc = pred_yolo[pred_yolo[:, 5] >= conf_threshold].copy()
    pred_voc[:, 1:5] = box_transform_xcycwh_to_x1y1x2y2(pred_voc[:, 1:5])
    pred_voc[:, 1:5] = scale_to_original(pred_voc[:, 1:5], scale_w=input_size, scale_h=input_size)
    canvas = visualize(canvas, pred_voc, class_list, color_list, show_class=True, show_score=True)
    return canvas[...,::-1]


def visualize_target(tensor_image, target, class_list, color_list):
    input_size = tensor_image.shape[-1]
    canvas = denormalize(tensor_image)
    target_voc = target.copy()
    target_voc[:, 1:5] = box_transform_xcycwh_to_x1y1x2y2(target_voc[:, 1:5])
    target_voc[:, 1:5] = scale_to_original(target_voc[:, 1:5], scale_w=input_size, scale_h=input_size)
    canvas = visualize(canvas, target_voc, class_list, color_list, show_class=True, show_score=False)
    return canvas[...,::-1]


def visualize_AP_per_class(data_df):
    plt.figure(figsize=(8, len(data_df)/4+4))
    ax = sns.barplot(x='AP_50', y='CATEGORY', data=data_df)
    ax.set(xlabel='Category', ylabel='AP@.50', title='AP@.50 per category')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def visualize_detect_rate_per_class(data_df, scale=10):
    df_melt = pd.melt(data_df.drop('AP_50', axis=1), id_vars='CATEGORY', var_name='SOURCE', value_name='VALUE')
    mask = df_melt.SOURCE.isin(['NUM_FP'])
    df_melt.loc[mask, 'VALUE'] /= scale

    plt.figure(figsize=(10, len(data_df)+2))
    ax = sns.barplot(x='VALUE', y='CATEGORY', hue='SOURCE', data=df_melt, palette='pastel')
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    ax.set_xlabel('FP')
    ax.set_xticklabels(ax.get_xticks() * scale)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels(ax.get_xticks())
    ax2.set_xlabel('True & TP & FN')
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def visualize_PR_curve_per_class(pr_pts_per_class, class_list):
    fig_PR_curves = {}
    for class_id in pr_pts_per_class.keys():
        mprec = pr_pts_per_class[class_id]['mprec'][::-1][1:]
        mrec = pr_pts_per_class[class_id]['mrec'][::-1][1:]
    
        plt.figure(figsize=(6,4))
        ax = sns.lineplot(x=mrec, y=mprec, estimator=None, sort=False)
        ax.set(xlabel='Recalls', ylabel='Precisions', title=f'{class_list[class_id]}')
        fig = ax.get_figure()
        fig.tight_layout()
        fig_PR_curves[class_id] = fig
    return fig_PR_curves
    


def analyse_mAP_info(mAP_info, class_list, areaRngLbl=('all', 'small', 'medium', 'large')):
    def sort_dict(values_per_class):
        return dict(sorted(values_per_class.items()))

    for areaLbl in areaRngLbl:
        AP_50_PER_CLASS = sort_dict(mAP_info[areaLbl]['AP_50_PER_CLASS'])
        NUM_TP_50_PER_CLASS = sort_dict(mAP_info[areaLbl]['NUM_TP_50_PER_CLASS'])
        NUM_FP_50_PER_CLASS = sort_dict(mAP_info[areaLbl]['NUM_FP_50_PER_CLASS'])
        NUM_TRUE_PER_CLASS = sort_dict(mAP_info[areaLbl]['NUM_TRUE_PER_CLASS'])
        PR_50_PTS_PER_CLASS = sort_dict(mAP_info[areaLbl]['PR_50_PTS_PER_CLASS'])

    data_dict = {}
    for class_id in AP_50_PER_CLASS.keys():
        data_dict[class_id] = [class_list[class_id],
                                AP_50_PER_CLASS[class_id],
                                NUM_TRUE_PER_CLASS[class_id],
                                NUM_TP_50_PER_CLASS[class_id],
                                NUM_TRUE_PER_CLASS[class_id] - NUM_TP_50_PER_CLASS[class_id],
                                NUM_FP_50_PER_CLASS[class_id]]

    data_df = pd.DataFrame.from_dict(data=data_dict, orient='index', columns=['CATEGORY', 'AP_50', 'NUM_TRUE', 'NUM_TP', 'NUM_FN', 'NUM_FP'])
    figure_AP = visualize_AP_per_class(data_df)
    fig_PR_curves = visualize_PR_curve_per_class(PR_50_PTS_PER_CLASS, class_list)
    figure_detect_rate = visualize_detect_rate_per_class(data_df, scale=10)
    
    return data_df, figure_AP, figure_detect_rate, fig_PR_curves