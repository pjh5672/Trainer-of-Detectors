import os
import torch
from tqdm import tqdm


def save_model(model, save_path):
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def build_progress_bar(dataloaders):
    progress_bar = {}
    for phase in ['train', 'val']:
        progress_bar[phase] = tqdm(iterable=dataloaders[phase],
                                   total=len(dataloaders[phase]),
                                   desc=f'[Phase:{phase.upper()}]',
                                   leave=False)
    return progress_bar