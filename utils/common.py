import os
import torch
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
                                   leave=False,
                                   ncols=200)
    return progress_bar


