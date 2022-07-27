import os
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm



class CacheMaker():
    def __init__(self, cache_dir, file_name, phase):
        if not cache_dir.is_dir():
            os.makedirs(cache_dir, exist_ok=True)
        self.phase = phase
        self.cache_path = cache_dir / f'{file_name}_{self.phase}.cache'
    
    
    def __call__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths
        
        try:
            cache = torch.load(self.cache_path)
            assert cache['hash'] == self.get_hash(self.image_paths + self.label_paths)
        except Exception as e:
            time_created = datetime.today().strftime('%Y-%m-%d_%H:%M')
            cache = self.cache_labels(self.cache_path, version=time_created)
        
        text = f'[{self.phase.upper()}] '
        for k in ('hash', 'version'):
            text += f'{k}: {cache.pop(k)}  '
        print(text)
        return cache
    
    
    def get_hash(self, files):
        return sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    

    def cache_labels(self, cache_path, version):
        pbar = tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths), ncols=200)
        
        x = {}
        for (image_path, label_path) in pbar:
            pbar.set_description(f'Caching Labels...')
            if os.path.isfile(label_path):
                with open(label_path, mode="r") as f:
                    item = [x.split() for x in f.read().splitlines()]
                    label = np.array(item, dtype=np.float32)
                x[image_path] = label

        x['hash'] = self.get_hash(self.image_paths + self.label_paths)
        x['version'] = version
        torch.save(x, cache_path)
        return x
