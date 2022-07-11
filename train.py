import os
from datetime import datetime
from collections import defaultdict

import yaml
import torch
from tqdm import tqdm

from dataloader import build_dataloader
from models import YOLOv3_Model
from loss_function import YOLOv3_Loss
from utils import *



class Trainer():
    def __init__(self, data_path, config_path, save_path):
        self.save_path = save_path
        self.time_created = datetime.today().strftime('%Y-%m-%d_%H-%M')
        os.makedirs(self.save_path, exist_ok=True)

        with open(config_path) as f:
            item = yaml.load(f, Loader=yaml.FullLoader)
        
        self.is_cuda = item['IS_CUDA']
        self.log_level = item['LOG_LEVEL']
        self.anchor_iou_threshold = item['ANCHOR_IOU_THRESHOLD']
        self.num_epochs = item['NUM_EPOCHS']
        self.input_size = item['INPUT_SIZE']
        self.batch_size = item['BATCH_SIZE']
        self.lr = item['LEARNING_RATE']
        self.weight_decay = item['WEIGHT_DECAY']

        self.logger = build_logger(log_path=self.save_path / 'logs', set_level=self.log_level)
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.is_cuda else 'cpu')
        self.dataloaders, self.classname_list = build_dataloader(data_path=data_path, 
                                                                image_size=(self.input_size, self.input_size), 
                                                                batch_size=self.batch_size)
        self.num_classes = len(self.classname_list)
        self.model = YOLOv3_Model(config_path=config_path, num_classes=self.num_classes, device=self.device)
        self.model = self.model.to(self.device)

        self.criterion = YOLOv3_Loss(config_path=config_path, model=self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataloader = tqdm(self.dataloaders['train'], desc='Calculating Best Possible Rate(BPR)...', ncols=200)
        PBR_params = [
            self.input_size, self.batch_size, 
            self.criterion.num_anchor_per_scale, 
            self.criterion.anchors, 
            self.criterion.strides
        ]
        message = f'Input Size: {self.input_size}'
        BPR_rate, total_n_anchor, total_n_target = check_best_possible_recall(dataloader, PBR_params, 
                                                                              self.anchor_iou_threshold)
        message += f', Best Possible Rate: {BPR_rate:0.5f}, Total_anchor/Total_target: {total_n_anchor}/{total_n_target}'
        self.logger.info(message)
        del dataloader
        

    def train_one_epoch(self):
        dataloader_pbars = build_progress_bar(self.dataloaders)
        loss_per_phase = defaultdict(float)
        loss_types = ['total', 'coord', 'obj', 'noobj', 'cls']

        for phase in ['train', 'val']:
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            
            for index, mini_batch in enumerate(dataloader_pbars[phase]):
                images = mini_batch[0].to(self.device, non_blocking=True)
                targets = mini_batch[1]
                filenames = mini_batch[2]
            
                with torch.set_grad_enabled(phase == 'train'):
                    predictions = self.model(images)
                losses = self.criterion(predictions, targets)

                if phase == 'train':
                    self.optimizer.zero_grad()
                    losses[0].backward()
                    self.optimizer.step()

                monitor_text = ''
                for loss_name, loss_value in zip(loss_types, losses):
                    loss_per_phase[f'{phase}_{loss_name}'] += loss_value.item()
                    monitor_text += f'{loss_name}: {loss_value.item():.2f} '
                dataloader_pbars[phase].set_postfix_str(s=f'{monitor_text}')

            for loss_name in loss_per_phase.keys():
                if loss_name.startswith(phase):
                    loss_per_phase[loss_name] /= len(dataloader_pbars[phase])
                    
            del mini_batch, losses
            torch.cuda.empty_cache()
        return loss_per_phase


    def run(self):
        best_score = float('inf')
        global_pbar = tqdm(range(self.num_epochs), ncols=200)
        
        for self.epoch in global_pbar:
            message = f'[Epoch:{self.epoch+1:02d}/{self.num_epochs}]'
            global_pbar.set_description(desc=message)
            loss_per_phase = self.train_one_epoch()

            monitor_text = f' Loss - Train: {loss_per_phase["train_total"]:.2f}, Val: {loss_per_phase["val_total"]:.2f}'
            self.logger.debug(message + monitor_text)
            
            if (self.epoch+1) % 10 == 0:
                save_model(model=self.model, 
                           save_path=self.save_path / 'weights',
                           model_name=f'{self.time_created}-{self.epoch+1:02d}.pth')
        global_pbar.close()



if __name__ == "__main__":
    from pathlib import Path

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]

    EXP_NAME = 'test'
    data_path = ROOT / 'data' / 'coco128.yml'
    config_path = ROOT / 'config' / 'yolov3.yml'
    save_path = ROOT / 'experiments' / EXP_NAME

    trainer = Trainer(data_path=data_path, config_path=config_path, save_path=save_path)
    trainer.run()