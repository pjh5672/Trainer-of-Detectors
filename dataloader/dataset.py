import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from transform import build_transformer
from utils import CacheMaker


class Dataset():
    def __init__(self, data_path, phase, transformer=None):

        with open(data_path) as f:
            data_item = yaml.load(f, Loader=yaml.FullLoader)
        
        self.data_dir = Path(data_item['PATH'])
        self.classname_list = data_item['NAMES']
        self.phase = phase
        
        self.image_paths = []
        for sub_dir in data_item[self.phase.upper()]:
            image_dir = self.data_dir / sub_dir
            self.image_paths += [str(image_dir / fn) for fn in os.listdir(image_dir) \
                                 if fn.lower().endswith(('.png','.jpg','.jpeg'))]
        self.label_paths = self.replace_image2label_paths(self.image_paths)
        
        cache_dir = Path(data_path).parent / 'caches'
        cache_name = Path(data_path).name.split('.')[0]
        cache_maker = CacheMaker(cache_dir, cache_name, phase)
        
        cache = cache_maker(self.image_paths, self.label_paths)
        self.image_paths = list(cache.keys())
        self.label_paths = self.replace_image2label_paths(self.image_paths)
        
        self.transformer = transformer


    def __len__(self): return len(self.image_paths)
    
    
    def __getitem__(self, index):
        filename, image = self.get_image(index)
        class_ids, bboxes, noobj_status = self.get_label(index)
        bboxes = self.clip_box_coordinates(bboxes)
        
        if self.transformer:
            transformed_data = self.transformer(image=image, bboxes=bboxes, class_ids=class_ids)
            image = transformed_data['image']
            bboxes = np.array(transformed_data['bboxes'])
            class_ids = np.array(transformed_data['class_ids'])

            if len(class_ids) == 0:
                class_ids = np.array([-1])
                bboxes = np.array([[0.5, 0.5, 1., 1.]], dtype=np.float32)
            
        target = np.concatenate((class_ids[:, np.newaxis], bboxes), axis=1)
        return image, target, filename
    
    
    def replace_image2label_paths(self, image_paths):
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in image_paths]
    
    
    def get_image(self, index):
        filename = self.image_paths[index].split(os.sep)[-1]
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return filename, image
    
        
    def get_label(self, index):
        with open(self.label_paths[index], mode="r") as f:
            item = [x.split() for x in f.read().splitlines()]
            label = np.array(item, dtype=np.float32)
        
        class_ids, bboxes, noobj_status = self.check_no_label(label)
        return class_ids, bboxes, noobj_status
    
    
    def check_no_label(self, label):
        if len(label) == 0:
            class_ids = np.array([-1])
            bboxes = np.array([[0.5, 0.5, 1., 1.]], dtype=np.float32)
            noobj_status = True
        else:
            class_ids = label[:, 0]
            bboxes = label[:, 1:5]
            noobj_status = False
        return class_ids, bboxes, noobj_status
    

    def clip_box_coordinates(self, bboxes):
        bboxes = self.box_transform_xcycwh_to_x1y1x2y2(bboxes)
        bboxes = self.box_transform_x1y1x2y2_to_xcycwh(bboxes)
        return bboxes
    

    def box_transform_xcycwh_to_x1y1x2y2(self, bboxes):
        x1y1 = bboxes[:, :2] - bboxes[:, 2:] / 2
        x2y2 = bboxes[:, :2] + bboxes[:, 2:] / 2
        x1y1x2y2 = np.concatenate((x1y1, x2y2), axis=1)
        x1y1x2y2 = x1y1x2y2.clip(min=0., max=1.)
        return x1y1x2y2
    

    def box_transform_x1y1x2y2_to_xcycwh(self, bboxes):
        wh = bboxes[:, 2:] - bboxes[:, :2]
        xcyc = bboxes[:, :2] + wh / 2
        xcycwh = np.concatenate((xcyc, wh), axis=1)
        return xcycwh

    @staticmethod
    def collate_fn(mini_batch):
        images = []
        targets = []
        filenames = []
        
        for image, target, filename in mini_batch:
            images.append(image)
            targets.append(target)
            filenames.append(filename)
        
        return torch.stack(images, dim=0), targets, filenames


def build_dataloader(data_path, image_size=(448, 448), batch_size=4):
    transformers = build_transformer(image_size=image_size)
    
    dataloaders = {}
    dset = Dataset(data_path=data_path, phase='train', transformer=transformers['train'])
    dataloaders['train'] = DataLoader(dset, batch_size=batch_size, shuffle=True, collate_fn=Dataset.collate_fn)

    dset = Dataset(data_path=data_path, phase='val', transformer=transformers['val'])
    dataloaders['val'] = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=Dataset.collate_fn)
    return dataloaders, dset.classname_list



if __name__ == '__main__':
    from pathlib import Path

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]

    data_path = ROOT / 'data' / 'coco128.yml'
    dataloaders, classname_list = build_dataloader(data_path=data_path, image_size=(448, 448), batch_size=4)

    # sanity check
    for _ in range(1):
        for phase in ['train', 'val']:
            for index, minibatch in enumerate(dataloaders[phase]):
                images = minibatch[0]
                targets = minibatch[1]
                filenames = minibatch[2]
                        
                if index % 30 == 0:
                    print(f"{phase} - {index}/{len(dataloaders[phase])}")