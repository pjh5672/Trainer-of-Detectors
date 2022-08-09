import os
from datetime import datetime

import torch
import numpy as np



def make_cache_file(cache_dir, file_name, phase, data_path, time_created):
    if not cache_dir.is_dir():
        os.makedirs(cache_dir, exist_ok=True)
    cache_path = cache_dir / f'{file_name}_{phase}.cache'
    try:
        cache = torch.load(cache_path)
        assert cache['hash'] == get_hash(data_path)
    except Exception as e:
        cache = cache_labels(cache_path, data_path, version=time_created)

    text = f'[{phase.upper()}] '
    for k in ('hash', 'version'):
        text += f'{k}: {cache.pop(k)} '
    return cache, text


def get_hash(files):
    if isinstance(files, list): 
        return sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    else:
        x = []
        for image_path, label_path in files:
            if os.path.isfile(label_path):
                x.append(image_path)
        return sum(os.path.getsize(f) for f in x if os.path.isfile(f))


def cache_labels(cache_path, data_path, version):
    x = {}
    for image_path, label_path in data_path:
        if not isinstance(data_path, zip):
            data_path.set_description(f'Caching Labels...[{cache_path.name}]')
        if os.path.isfile(label_path):
            with open(label_path, mode="r") as f:
                item = [x.split() for x in f.read().splitlines()]
                label = np.array(item, dtype=np.float32)
            x[image_path] = label

    x['hash'] = get_hash(list(x.keys()))
    x['version'] = version
    torch.save(x, cache_path)
    return x