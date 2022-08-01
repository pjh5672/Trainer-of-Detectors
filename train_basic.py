import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import yaml
import torch
from tqdm import tqdm

from dataloader import build_dataloader
from models import YOLOv3_Model
from loss_function import YOLOv3_Loss
from utils import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]



class Trainer():
    def __init__(self, args, timestamp):
        with open(args.data_path) as f:
            data_item = yaml.load(f, Loader=yaml.FullLoader)
        with open(args.config_path) as f:
            config_item = yaml.load(f, Loader=yaml.FullLoader)

        self.time_created = timestamp
        self.is_cuda = config_item['IS_CUDA']
        self.log_level = config_item['LOG_LEVEL']
        self.anchor_iou_threshold = config_item['ANCHOR_IOU_THRESHOLD']
        self.min_conf_threshold = config_item['MIN_SCORE_THRESH']
        self.min_iou_threshold = config_item['MIN_IOU_THRESH']
        self.num_epochs = config_item['NUM_EPOCHS']
        self.input_size = config_item['INPUT_SIZE']
        self.batch_size = config_item['BATCH_SIZE']
        self.lr = config_item['LEARNING_RATE']
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.is_cuda else 'cpu')
        self.dataloaders, self.class_list = build_dataloader(data_path=args.data_path, 
                                                                image_size=(self.input_size, self.input_size), 
                                                                batch_size=self.batch_size)
        self.val_file = args.data_path.parent / data_item['mAP_FILE']
        assert self.val_file.is_file(), RuntimeError(f'Not exist val file, expected {self.val_file}')

        self.num_classes = len(self.class_list)
        self.model = YOLOv3_Model(config_path=args.config_path, num_classes=self.num_classes)
        self.criterion = YOLOv3_Loss(config_path=args.config_path, model=self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.evaluator = Evaluator(GT_file=self.val_file, model_input_size=self.input_size)
        self.model = self.model.to(self.device)
        self.color_list = generate_random_color(num_colors=self.num_classes)

        dataloader = tqdm(self.dataloaders['train'], desc='Calculating Best Possible Rate(BPR)...', ncols=200)
        PBR_params = [self.input_size, self.criterion.num_anchor_per_scale, self.criterion.anchors, self.criterion.strides]
        BPR_rate, total_n_anchor, total_n_target = check_best_possible_recall(dataloader, PBR_params, self.anchor_iou_threshold)
        
        self.logger = build_logger(log_path=args.exp_path / 'logs', log_name=f'{self.time_created}.log', set_level=self.log_level)
        message = f'Input Size: {self.input_size}'
        self.logger.info(message)
        message += f'Best Possible Rate: {BPR_rate:0.4f}, Total_anchor/Total_target: {total_n_anchor}/{total_n_target}'
        self.logger.info(message)
        del dataloader
        

    def train_one_epoch(self):
        dataloader_pbars = build_progress_bar(self.dataloaders)
        loss_per_phase = defaultdict(float)
        loss_types = ['total', 'coord', 'obj', 'noobj', 'cls']
        detections = []

        for phase in ['train', 'val']:
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            
            for index, mini_batch in enumerate(dataloader_pbars[phase]):
                if index == 0:
                    canvas = mini_batch[0][0]

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
                
                elif phase == 'val':
                    for idx in range(len(filenames)):
                        filename = filenames[idx]
                        pred_yolo = torch.cat(predictions, dim=1)[idx].cpu().numpy()
                        pred_yolo = filter_obj_score(prediction=pred_yolo, conf_threshold=self.min_conf_threshold)
                        pred_yolo = run_NMS_for_yolo(prediction=pred_yolo, iou_threshold=self.min_iou_threshold)
                        if len(pred_yolo) > 0:
                            detections.append((filename, pred_yolo))

                monitor_text = ''
                for loss_name, loss_value in zip(loss_types, losses):
                    loss_per_phase[f'{phase}_{loss_name}'] += loss_value.item()
                    monitor_text += f'{loss_name}: {loss_value.item():.2f} '
                dataloader_pbars[phase].set_postfix_str(s=f'{monitor_text}')

            for loss_name in loss_per_phase.keys():
                if loss_name.startswith(phase):
                    loss_per_phase[loss_name] /= len(dataloader_pbars[phase])

            if phase == 'val':
                filename, pred_yolo = detections[0]
                canvas = visualize_prediction(self.evaluator.image_to_info, 
                                                canvas, filename, pred_yolo, 
                                                self.class_list, self.color_list)
                mAP_info, eval_text = self.evaluator(detections)
            del mini_batch, losses
            torch.cuda.empty_cache()

        return loss_per_phase, mAP_info, eval_text, canvas


    def run(self, args):
        best_mAP = 0.01
        global_pbar = tqdm(range(1, self.num_epochs+1), ncols=200)
        
        for epoch in global_pbar:
            message = f'[Epoch:{epoch:02d}/{self.num_epochs:03d}]'
            global_pbar.set_description(desc=message)
            loss_per_phase, mAP_info, eval_text, canvas = self.train_one_epoch()

            monitor_text = f' Train Loss: {loss_per_phase["train_total"]:.2f}, Val Loss: {loss_per_phase["val_total"]:.2f}'
            self.logger.debug(message + monitor_text)
            self.logger.info(eval_text)

            if (epoch) % args.img_log_interval == 0:
                imwrite(str(args.exp_path / 'images' / f'{self.time_created}-EP{epoch:02d}.jpg'), canvas)

            if mAP_info['all']['mAP05'] > best_mAP:
                best_mAP = mAP_info['all']['mAP05']
                save_model(model=self.model, save_path=args.exp_path / 'weights', model_name=f'{self.time_created}-EP{epoch:02d}.pth')

        global_pbar.close()



def main():
    time_created = datetime.today().strftime('%Y-%m-%d_%H-%M')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/coco128.yml', help='Path to data.yml file')
    parser.add_argument('--config_path', type=str, default='config/yolov3.yml', help='Path to config.yml file')
    parser.add_argument('--exp_name', type=str, default=str(time_created), help='Name to log training')
    parser.add_argument('--img_log_interval', type=int, default=5, help='Image logging interval')
    args = parser.parse_args()
    
    args.data_path = ROOT / args.data_path
    args.config_path = ROOT / args.config_path
    args.exp_path = ROOT / 'experiments' / args.exp_name
    os.makedirs(args.exp_path / 'logs', exist_ok=True)
    os.makedirs(args.exp_path / 'images', exist_ok=True)
    trainer = Trainer(args=args, timestamp=time_created)
    trainer.run()

if __name__ == "__main__":
    main()