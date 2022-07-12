import os
import torch
import numpy as np
from tqdm import tqdm



def save_model(model, save_path, model_name):
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path / model_name)


def build_progress_bar(dataloaders):
    progress_bar = {}
    for phase in ['train', 'val']:
        progress_bar[phase] = tqdm(iterable=dataloaders[phase],
                                   total=len(dataloaders[phase]),
                                   desc=f'[Phase:{phase.upper()}]',
                                   leave=False, ncols=200)
    return progress_bar


def get_IoU_target_with_anchor(wh1, wh2):
    w1, h1 = wh1
    w2, h2 = wh2
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area


def check_best_possible_recall(dataloader, PBR_params, anchor_iou_threshold=0.25):
    input_size, num_anchor_per_scale, anchors, strides = PBR_params
    anchors = [anchor.cpu() for anchor in anchors]
    total_n_target, total_n_anchor = 0, 0

    for index, mini_batch in enumerate(dataloader):
        targets = mini_batch[1]
        n_target_per_mini_batch, n_anchor_per_mini_batch = 0, 0

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
            
            n_target_per_mini_batch += len(target)
            n_anchor_per_mini_batch += len(best_possible_anchor)
        
        total_n_target += n_target_per_mini_batch
        total_n_anchor += n_anchor_per_mini_batch
    
    BPR_rate = total_n_anchor/total_n_target
    return BPR_rate, total_n_anchor, total_n_target