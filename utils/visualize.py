import random
import cv2
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from general import box_transform_xcycwh_to_x1y1x2y2, scale_to_original

matplotlib.use('Agg')
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


def visualize_bbox(image, label, class_list, color_list, show_class=False, show_score=False, fontscale=0.7, thickness=2):
    class_id = int(label[0])
    if class_id > 0:
        color = color_list[class_id]
        x_min, y_min, x_max, y_max = list(map(int, label[1:5]))
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


def visualize_prediction(tensor_image, detection, conf_threshold, class_list, color_list):
    input_size = tensor_image.shape[-1]
    canvas = denormalize(tensor_image)
    pred_voc = detection[detection[:, 5] >= conf_threshold].copy()
    if len(pred_voc) > 0:
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


def show_values(axs, orient='h', space=0.005, mode='ap'):
    def _single(ax):
        if orient == 'v':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                if mode == 'ap':
                    value = f'{p.get_height():.2f}'
                else:
                    value = f'{p.get_height():.0f}'
                ax.text(_x, _y, value, ha='center') 
        elif orient == 'h':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.3)
                if mode == 'ap':
                    value = f'{p.get_width():.2f}'
                else:
                    value = f'{p.get_width():.0f}'
                ax.text(_x, _y, value, ha='left')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def visualize_AP_per_class(data_df):
    plt.figure(figsize=(8, len(data_df)/4+4))
    ax = sns.barplot(x='AP_50', y='CATEGORY', data=data_df)
    ax.set(xlabel='AP@.50', ylabel='Category', title=f'AP@0.5 per categories (mAP@0.5: {data_df["AP_50"].mean():.4f})')
    show_values(ax, 'h', mode='ap')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    plt.grid('on')
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def visualize_detect_rate_per_class(data_df):
    df_melt = pd.melt(data_df.drop('AP_50', axis=1), id_vars='CATEGORY', var_name='SOURCE', value_name='VALUE')
    plt.figure(figsize=(10, len(data_df)+2))
    ax = sns.barplot(x='VALUE', y='CATEGORY', hue='SOURCE', data=df_melt, palette='pastel')
    ax.set(xlabel='Count', ylabel='Category', title='Groundtruth & Detection')
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    show_values(ax, 'h', mode=None)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    plt.grid('on')
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def visualize_PR_curve_per_class(pr_pts_per_class, class_list):
    fig_PR_curves = {}
    
    with plt.rc_context(rc={'figure.max_open_warning': 0}):
        for class_id in pr_pts_per_class.keys():
            mprec = pr_pts_per_class[class_id]['mprec'][::-1][1:]
            mrec = pr_pts_per_class[class_id]['mrec'][::-1][1:]
            
            plt.figure(figsize=(4, 4))
            ax = sns.lineplot(x=mrec, y=mprec, estimator=None, sort=False)
            ax.set(xlabel='Recalls', ylabel='Precisions')
            ax.set_title(label=f'Category: {class_list[class_id]}', fontsize=12)
            plt.grid('on')
            fig = ax.get_figure()
            fig.tight_layout()
            fig_PR_curves[class_id] = fig
    return fig_PR_curves
    

def sort_dict(values_per_class):
    return dict(sorted(values_per_class.items()))


def analyse_mAP_info(mAP_info_area, class_list):
    AP_50_PER_CLASS = sort_dict(mAP_info_area['AP_50_PER_CLASS'])
    NUM_TP_50_PER_CLASS = sort_dict(mAP_info_area['NUM_TP_50_PER_CLASS'])
    NUM_FP_50_PER_CLASS = sort_dict(mAP_info_area['NUM_FP_50_PER_CLASS'])
    NUM_TRUE_PER_CLASS = sort_dict(mAP_info_area['NUM_TRUE_PER_CLASS'])
    PR_50_PTS_PER_CLASS = sort_dict(mAP_info_area['PR_50_PTS_PER_CLASS'])

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
    figure_detect_rate = visualize_detect_rate_per_class(data_df)
    return data_df, figure_AP, figure_detect_rate, fig_PR_curves