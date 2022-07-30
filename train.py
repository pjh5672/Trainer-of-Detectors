import os
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



class Trainer():
    def __init__(self, data_path, config_path, save_path):
        self.save_path = save_path
        self.time_created = datetime.today().strftime('%Y-%m-%d_%H-%M')
        os.makedirs(self.save_path, exist_ok=True)

        with open(data_path) as f:
            data_item = yaml.load(f, Loader=yaml.FullLoader)

        with open(config_path) as f:
            config_item = yaml.load(f, Loader=yaml.FullLoader)
        
        self.is_cuda = config_item['IS_CUDA']
        self.log_level = config_item['LOG_LEVEL']
        self.anchor_iou_threshold = config_item['ANCHOR_IOU_THRESHOLD']
        self.min_conf_threshold = config_item['MIN_SCORE_THRESH']
        self.min_iou_threshold = config_item['MIN_IOU_THRESH']
        self.num_epochs = config_item['NUM_EPOCHS']
        self.input_size = config_item['INPUT_SIZE']
        self.batch_size = config_item['BATCH_SIZE']
        self.lr = config_item['LEARNING_RATE']
        self.weight_decay = config_item['WEIGHT_DECAY']

        self.img_log_dir = self.save_path / 'images'
        os.makedirs(self.img_log_dir, exist_ok=True)

        self.logger = build_logger(log_path=self.save_path / 'logs', set_level=self.log_level)
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.is_cuda else 'cpu')
        self.dataloaders, self.classname_list = build_dataloader(data_path=data_path, 
                                                                image_size=(self.input_size, self.input_size), 
                                                                batch_size=self.batch_size)
        self.val_file = data_path.parent / data_item['mAP_FILE']
        assert self.val_file.is_file(), RuntimeError(f'Not exist val file, expected {self.val_file}')

        self.num_classes = len(self.classname_list)
        self.model = YOLOv3_Model(config_path=config_path, num_classes=self.num_classes, device=self.device)
        self.model = self.model.to(self.device)

        self.criterion = YOLOv3_Loss(config_path=config_path, model=self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.evaluator = Evaluator(GT_file=self.val_file, model_input_size=self.input_size)
        self.color_list = generate_random_color(num_colors=self.num_classes)

        dataloader = tqdm(self.dataloaders['train'], desc='Calculating Best Possible Rate(BPR)...', ncols=200)
        PBR_params = [
            self.input_size,
            self.criterion.num_anchor_per_scale, 
            self.criterion.anchors, 
            self.criterion.strides
        ]
        message = f'Input Size: {self.input_size}'
        BPR_rate, total_n_anchor, total_n_target = check_best_possible_recall(dataloader, PBR_params, self.anchor_iou_threshold)
        message += f', Best Possible Rate: {BPR_rate:0.5f}, Total_anchor/Total_target: {total_n_anchor}/{total_n_target}'
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
                    for idx in range(self.batch_size):
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
                img_h = self.evaluator.image_to_info[filename]['height']
                img_w = self.evaluator.image_to_info[filename]['width']
                canvas = denormalize(canvas)
                canvas = cv2.resize(canvas, dsize=(img_w, img_h))
                pred_voc = pred_yolo.copy()
                pred_voc[:, 1:5] = box_transform_xcycwh_to_x1y1x2y2(pred_voc[:, 1:5]/self.input_size)
                pred_voc[:, 1:5] = scale_to_original(pred_voc[:, 1:5], scale_w=img_w, scale_h=img_h)
                canvas = visualize(canvas, pred_voc, self.classname_list, self.color_list, show_class=True, show_score=True)
                mAP_info, eval_text = self.evaluator(detections)

            del mini_batch, losses
            torch.cuda.empty_cache()

        return loss_per_phase, mAP_info, eval_text, canvas


    def run(self):
        best_mAP = 0.01
        global_pbar = tqdm(range(self.num_epochs), ncols=200)
        
        for epoch in global_pbar:
            message = f'[Epoch:{epoch+1:02d}/{self.num_epochs}]'
            global_pbar.set_description(desc=message)
            loss_per_phase, mAP_info, eval_text, canvas = self.train_one_epoch()

            monitor_text = f' Loss - Train: {loss_per_phase["train_total"]:.2f}, Val: {loss_per_phase["val_total"]:.2f}'
            self.logger.debug(message + monitor_text)
            self.logger.info(eval_text)

            if (epoch+1) % 2 == 0:
                imwrite(str(self.img_log_dir.resolve() / f'{self.time_created}-EP{epoch+1:02d}.jpg'), canvas)

            if mAP_info['all']['mAP05'] > best_mAP:
                best_mAP = mAP_info['all']['mAP05']
                save_model(model=self.model, 
                           save_path=self.save_path / 'weights',
                           model_name=f'{self.time_created}-EP{epoch+1:02d}.pth')
        global_pbar.close()



if __name__ == "__main__":
    from pathlib import Path

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]

    EXP_NAME = 'test13'
    data_path = ROOT / 'data' / 'coco128.yml'
    config_path = ROOT / 'config' / 'yolov3.yml'
    save_path = ROOT / 'experiments' / EXP_NAME

    trainer = Trainer(data_path=data_path, config_path=config_path, save_path=save_path)
    trainer.run()