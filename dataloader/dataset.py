import os
import sys
import json
from pathlib import Path
from datetime import datetime

import yaml
import cv2
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from transform import build_transformer
from utils import *



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
                                 if fn.lower().endswith(('png','jpg','jpeg'))]
        self.label_paths = self.replace_image2label_paths(self.image_paths)
        self.generate_no_label(self.label_paths)

        GT_dir = Path(data_path).parent / 'evals'
        cache_dir = Path(data_path).parent / 'caches'
        save_name = Path(data_path).name.split('.')[0]
        
        if phase == 'val':
            self.generate_GT_for_mAP(save_dir=GT_dir, file_name=save_name, phase=phase)
            
        cache_maker = CacheMaker(cache_dir=cache_dir, file_name=save_name, phase=phase)
        cache = cache_maker(self.image_paths, self.label_paths)
        assert len(self.image_paths) == len(list(cache.keys())), "Not match loaded files wite cache files" 
        
        self.transformer = transformer


    def __len__(self): return len(self.image_paths)
    
    
    def __getitem__(self, index):
        filename, image = self.get_image(index)
        class_ids, bboxes = self.get_label(index)
         
        if self.transformer:
            transformed_data = self.transformer(image=image, bboxes=bboxes, class_ids=class_ids)
            image = transformed_data['image']
            bboxes = np.array(transformed_data['bboxes'], dtype=np.float32)
            class_ids = np.array(transformed_data['class_ids'], dtype=np.float32)

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
        
        class_ids, bboxes = self.check_no_label(label)
        bboxes = self.clip_box_coordinates(bboxes)
        return class_ids, bboxes
    
        
    def generate_no_label(self, label_paths):
        for label_path in label_paths:
            if not os.path.isfile(label_path):
                f = open(str(label_path), mode='w')
                f.close()


    def check_no_label(self, label):
        if len(label) == 0:
            class_ids = np.array([-1])
            bboxes = np.array([[0.5, 0.5, 1., 1.]], dtype=np.float32)
        else:
            class_ids = label[:, 0]
            bboxes = label[:, 1:5]
        return class_ids, bboxes
    

    def clip_box_coordinates(self, bboxes):
        bboxes = box_transform_xcycwh_to_x1y1x2y2(bboxes)
        bboxes = box_transform_x1y1x2y2_to_xcycwh(bboxes)
        return bboxes


    def generate_GT_for_mAP(self, save_dir, file_name, phase):
        if not save_dir.is_dir():
            os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f'{file_name}_{phase}.json'

        if not save_path.is_file():
            eval_data = {}
            eval_data['images'] = []
            eval_data['annotations'] = []
            eval_data['categories'] = {}
            eval_data['timestamp'] = datetime.today().strftime('%Y-%m-%d_%H:%M')
            
            pbar = tqdm(range(len(self.label_paths)), ncols=200)
            img_id = 0
            anno_id = 0
            for index in pbar:
                pbar.set_description(f'Generating GT file for mAP evaluation...')
                filename, image = self.get_image(index)
                class_ids, bboxes = self.get_label(index)
                bboxes = box_transform_xcycwh_to_x1y1x2y2(bboxes)

                height, width, _ = image.shape
                anno_bboxes = bboxes.copy()
                anno_bboxes[:, [0,2]] *= width
                anno_bboxes[:, [1,3]] *= height

                for class_id, anno_bbox in zip(class_ids, anno_bboxes):
                    lbl_dict = {}
                    lbl_dict['id'] = anno_id
                    lbl_dict['image_id'] = img_id
                    lbl_dict['bbox'] = [round(pt, 2) for pt in list(map(float, anno_bbox))]
                    lbl_dict['area'] = round(float((anno_bbox[2]-anno_bbox[0])*(anno_bbox[3]-anno_bbox[1])),2)
                    lbl_dict['class_id'] = int(class_id)
                    eval_data['annotations'].append(lbl_dict)
                    anno_id += 1
                
                img_dict = {}
                img_dict['id'] = img_id
                img_dict['filename'] = filename
                img_dict['height'] = height
                img_dict['width'] = width
                eval_data['images'].append(img_dict)
                img_id += 1

            for idx in range(len(self.classname_list)):
                eval_data['categories'][idx] = self.classname_list[idx]

            with open(save_path, 'w') as outfile:
                json.dump(eval_data, outfile)     


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



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]

    data_path = ROOT / 'data' / 'coco128.yml'
    transformers = build_transformer(image_size=(416, 416))
    train_dset = Dataset(data_path=data_path, phase='train', transformer=transformers['train'])
    val_dset = Dataset(data_path=data_path, phase='val', transformer=transformers['val'])
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dset, batch_size=batch_size, collate_fn=Dataset.collate_fn, pin_memory=True)
    dataloaders['val'] = DataLoader(val_dset, batch_size=batch_size, collate_fn=Dataset.collate_fn, pin_memory=True)         

    # sanity check
    for _ in range(1):
        for phase in ['train', 'val']:
            for index, minibatch in enumerate(dataloaders[phase]):
                images = minibatch[0]
                targets = minibatch[1]
                filenames = minibatch[2]
                        
                if index % 30 == 0:
                    print(f"{phase} - {index}/{len(dataloaders[phase])}")