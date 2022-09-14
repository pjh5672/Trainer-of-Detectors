import os
import sys
import json
import random
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

from transform import *
from utils import *



class Dataset():
    def __init__(self, args, phase, rank, time_created):
        with open(args.data_path, mode='r') as f:
            self.data_item = yaml.load(f, Loader=yaml.FullLoader)
        with open(args.config_path, mode='r') as f:
            self.config_item = yaml.load(f, Loader=yaml.FullLoader)
        
        self.data_dir = Path(self.data_item['PATH'])
        self.classname_list = self.data_item['NAMES']
        self.input_size = self.config_item['INPUT_SIZE']
        self.phase = phase
        
        self.image_paths = []
        for sub_dir in self.data_item[self.phase.upper()]:
            image_dir = self.data_dir / sub_dir
            self.image_paths += [str(image_dir / fn) for fn in os.listdir(image_dir) if fn.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.label_paths = self.replace_image2label_paths(self.image_paths)
        self.generate_no_label(self.label_paths)

        GT_dir = Path(args.data_path).parent / 'evals'
        cache_dir = Path(args.data_path).parent / 'caches'
        save_name = Path(args.data_path).name.split('.')[0]
         
        if self.phase == 'val':
            self.generate_GT_for_mAP(save_dir=GT_dir, file_name=save_name, phase=phase, rank=rank)

        if rank == 0:
            data_path = tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths), ncols=115)
        else:
            data_path = zip(self.image_paths, self.label_paths)

        cache, self.data_info = make_cache_file(cache_dir=cache_dir, file_name=save_name, phase=phase, data_path=data_path, time_created=time_created)
        assert len(self.image_paths) == len(list(cache.keys())), "Not match loaded files wite cache files"
        self.album_transform = Albumentations(p_flipud=self.config_item['FLIP_UD'], p_fliplr=self.config_item['FLIP_LR'])


    def __len__(self): return len(self.image_paths)
    
    
    def __getitem__(self, index):
        filename, image, target = self.get_item(index)
        if self.phase == 'train':
            max_size = 0
            image, target = self.augment_item(image=image, target=target)
            if random.random() <= self.config_item['MIXUP']:
                _, image2, target2 = self.get_item(random.randint(0, len(self)-1))
                image2, target2 = self.augment_item(image=image2, target=target2)
                image, target = mixup(image1=image, target1=target, image2=image2, target2=target2)
            if len(target) == 0:
                target = np.array([[-1, 0., 0., self.input_size, self.input_size]], dtype=np.float32)
            target[:, 1:5] = scale_to_norm(target[:, 1:5], image_w=self.input_size, image_h=self.input_size)
            target[:, 1:5] = box_transform_x1y1x2y2_to_xcycwh(target[:, 1:5])
        else:
            image, target[:, 1:5], max_size = transform_square_image(image, target[:, 1:5])
            image = cv2.resize(image, dsize=(self.input_size, self.input_size))
        tensor = normalize(to_tensor(image))
        return tensor, target, filename, max_side
    

    def replace_image2label_paths(self, image_paths):
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in image_paths]
    
    
    def get_item(self, index):
        filename, image = self.get_image(index)
        class_ids, bboxes = self.get_label(index)
        bboxes = clip_box_coordinates(bboxes)
        target = np.concatenate((class_ids[:, np.newaxis], bboxes), axis=1)
        return filename, image, target


    def augment_item(self, image, target):
        img_h, img_w, _ = image.shape
        image, target = self.album_transform(image=image, target=target)
        image = augment_hsv(image, 
                            hgain=self.config_item['HSV_H'], 
                            sgain=self.config_item['HSV_S'], 
                            vgain=self.config_item['HSV_V'])
        target[:, 1:5] = box_transform_xcycwh_to_x1y1x2y2(target[:, 1:5])
        target[:, 1:5] = scale_to_original(target[:, 1:5], scale_w=img_w, scale_h=img_h)
        image, target = random_perspective(image, target,
                                           input_size=self.input_size,
                                           degrees=self.config_item['ROTATE'], 
                                           translate=self.config_item['SHIFT'], 
                                           scale=self.config_item['SCALE'], 
                                           perspective=self.config_item['PERSPECTIVE'])
        image, target = random_crop(image, target,
                                    x_max=img_w if img_w > self.input_size else self.input_size, 
                                    y_max=img_h if img_h > self.input_size else self.input_size, 
                                    crop_size=self.input_size)
        idx = box_candidates(box1=target[:, 1:5].T, wh_thr=4, ar_thr=20)
        return image, target[idx]


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
    

    def generate_GT_for_mAP(self, save_dir, file_name, phase, rank):
        if not save_dir.is_dir():
            os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f'{file_name}_{phase}.json'

        if not save_path.is_file():
            eval_data = {}
            eval_data['images'] = []
            eval_data['annotations'] = []
            eval_data['categories'] = {}
            eval_data['timestamp'] = datetime.today().strftime('%Y-%m-%d_%H:%M')
            
            pbar = tqdm(range(len(self.label_paths)), ncols=115) if rank == 0 else range(len(self.label_paths))
            img_id = 0
            anno_id = 0
            for index in pbar:
                if rank == 0:
                    pbar.set_description(f'Generating [{phase.upper()}] GT file for mAP evaluation...')

                filename, image = self.get_image(index)
                class_ids, bboxes = self.get_label(index)
                bboxes = clip_box_coordinates(bboxes)
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
                    lbl_dict['area'] = round(float((anno_bbox[2]-anno_bbox[0]+1)*(anno_bbox[3]-anno_bbox[1]+1)),2)
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
        max_sides = []
        
        for image, target, filename, max_side in mini_batch:
            images.append(image)
            targets.append(target)
            filenames.append(filename)
            max_sides.append(max_side)
        return torch.stack(images, dim=0), targets, filenames, max_sides



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]

    class args:
        data_path = ROOT / 'data' / 'coco128.yaml'
        config_path = ROOT / 'config' / 'yolov3_coco.yaml'
    train_dataset = Dataset(args=args, phase='train', rank=0, time_created=0)
    val_dataset = Dataset(args=args, phase='val', rank=0, time_created=0)
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=1, collate_fn=Dataset.collate_fn, pin_memory=True)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=1, collate_fn=Dataset.collate_fn, pin_memory=True)         

    # sanity check
    for _ in range(1):
        for phase in ['train', 'val']:
            for index, minibatch in enumerate(dataloaders[phase]):
                images = minibatch[0]
                targets = minibatch[1]
                filenames = minibatch[2]
                max_sides = minibatch[3]
    
                if index % 30 == 0:
                    print(f"{phase} - {index}/{len(dataloaders[phase])} - {max_sides}")