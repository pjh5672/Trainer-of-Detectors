import os
import shutil
import logging
import platform
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import cv2
import yaml
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataloader import Dataset, build_transformer
from models import YOLOv3_Model
from loss_function import YOLOv3_Loss
from utils import *


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
OS_SYSTEM = platform.system()
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')
cudnn.benchmark = True

assert OS_SYSTEM in ('Linux', 'Windows'), 'This is not supported on the Operating System.'



def setup(rank, world_size):
    if OS_SYSTEM == 'Linux':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        # initialize the process group
        dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(seconds=3600))


def cleanup():
    if OS_SYSTEM == 'Linux':
        dist.destroy_process_group()
    

def execute_train(rank, dataloader, model, criterion, optimizer):
    loss_per_phase = defaultdict(float)
    loss_types = ['total', 'coord', 'obj', 'noobj', 'cls']
    model.train()

    for index, mini_batch in enumerate(dataloader):
        images = mini_batch[0].cuda(rank, non_blocking=True) 
        targets = mini_batch[1]
        filenames = mini_batch[2]

        with torch.set_grad_enabled(True):
            predictions = model(images)

        losses = criterion(predictions, targets)
        optimizer.zero_grad()
        losses[0].backward()
        optimizer.step()

        monitor_text = ''
        for loss_name, loss_value in zip(loss_types, losses):
            loss_per_phase[f'{loss_name}'] += loss_value
            monitor_text += f'{loss_name}: {loss_value.item():.2f} '
        if rank == 0:
            dataloader.set_postfix_str(s=f'{monitor_text}')
    
    for loss_name in loss_per_phase.keys():
        loss_per_phase[loss_name] /= len(dataloader)
        if OS_SYSTEM == 'Linux':
            dist.all_reduce(loss_per_phase[loss_name], op=dist.ReduceOp.SUM)

    del mini_batch, losses
    torch.cuda.empty_cache()
    return loss_per_phase
    

@torch.no_grad()
def execute_val(rank, world_size, config, dataloader, model, criterion, evaluator, class_list, color_list):
    loss_per_phase = defaultdict(float)
    loss_types = ['total', 'coord', 'obj', 'noobj', 'cls']
    gather_objects = [None, ] * world_size
    detections = []
    model.eval()

    for index, mini_batch in enumerate(dataloader):
        if index == 0:
            canvas = mini_batch[0][0]
        images = mini_batch[0].cuda(rank, non_blocking=True) 
        targets = mini_batch[1]
        filenames = mini_batch[2]
        max_sides = mini_batch[3]

        predictions = model(images)
        losses = criterion(predictions, targets)

        for idx in range(len(filenames)):
            filename = filenames[idx]
            max_side = max_sides[idx]
            pred_yolo = torch.cat(predictions, dim=1)[idx].cpu().numpy()
            pred_yolo[:, :4] = clip_box_coordinates(bboxes=pred_yolo[:, :4]/config['INPUT_SIZE'])
            pred_yolo = filter_obj_score(prediction=pred_yolo, conf_threshold=config['MIN_SCORE_THRESH'])
            pred_yolo = run_NMS_for_YOLO(prediction=pred_yolo, iou_threshold=config['MIN_IOU_THRESH'], multi_label=True, maxDets=config['MAX_DETS'])
            
            if len(pred_yolo) > 0:
                detections.append((filename, pred_yolo, max_side))

        monitor_text = ''
        for loss_name, loss_value in zip(loss_types, losses):
            loss_per_phase[f'{loss_name}'] += loss_value
            monitor_text += f'{loss_name}: {loss_value.item():.2f} '

        if rank == 0:
            dataloader.set_postfix_str(s=f'{monitor_text}')

    for loss_name in loss_per_phase.keys():
        loss_per_phase[loss_name] /= len(dataloader)
        if OS_SYSTEM == 'Linux':
            dist.all_reduce(loss_per_phase[loss_name], op=dist.ReduceOp.SUM)
    if OS_SYSTEM == 'Linux':
        dist.all_gather_object(gather_objects, detections)
    elif OS_SYSTEM == 'Windows':
        gather_objects = [detections]

    canvas = visualize_prediction(canvas, detections[0], config['MIN_SCORE_THRESH_FOR_IMAGING'], class_list, color_list)
    del mini_batch, losses
    torch.cuda.empty_cache()
    return loss_per_phase, gather_objects, canvas


def main_work(rank, world_size, args, logger):
    ################################### Init Params ###################################
    setup_worker_logging(rank, logger)
    shutil.copy2(args.data_path, args.exp_path / args.data_path.name)
    shutil.copy2(args.config_path, args.exp_path / args.config_path.name)

    with open(args.data_path, mode='r') as f:
        data_item = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.config_path, mode='r') as f:
        config_item = yaml.load(f, Loader=yaml.FullLoader)

    input_size = config_item['INPUT_SIZE']
    class_list = data_item['NAMES']
    color_list = generate_random_color(num_colors=len(class_list))
    transformers = build_transformer(input_size=(input_size, input_size), augment_strong=config_item['AUGMENT_STRONG'])
    train_set = Dataset(data_path=args.data_path, phase='train', rank=rank, time_created=TIMESTAMP, transformer=transformers['train'], 
                        augment_infos=(transformers['input_size'], transformers['augment_strong']))
    val_set = Dataset(data_path=args.data_path, phase='val',  rank=rank, time_created=TIMESTAMP, transformer=transformers['val'])
    
    if rank == 0:
        logging.warning(f'{train_set.data_info}')
        logging.warning(f'{val_set.data_info}')
        print(f'{train_set.data_info}| {val_set.data_info}')

    model = YOLOv3_Model(config_path=args.config_path, num_classes=len(class_list))
    criterion = YOLOv3_Loss(config_path=args.config_path, model=model)
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=config_item['LEARNING_RATE'], weight_decay=config_item['WEIGHT_DECAY'])
    val_file = args.data_path.parent / data_item['mAP_FILE']
    assert val_file.is_file(), RuntimeError(f'Not exist val file, expected {val_file}')
    evaluator = Evaluator(GT_file=val_file, config=config_item)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config_item['LR_ADJUST_EPOCH'], gamma=0.1)
    
    ################################### Init Process ###################################
    setup(rank, world_size)

    ################################### Init Loader ####################################
    train_sampler = distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = distributed.DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(dataset=train_set, collate_fn=Dataset.collate_fn, batch_size=int(config_item['BATCH_SIZE']/world_size), 
                             shuffle=False, num_workers=world_size*4, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_set, collate_fn=Dataset.collate_fn, batch_size=int(config_item['BATCH_SIZE']/world_size),
                            shuffle=False, num_workers=world_size*4, pin_memory=True, sampler=val_sampler)

    ################################### Calculate BPR ####################################
    if config_item['GET_PBR']:
        dataloader = tqdm(train_loader, desc='Calculating Best Possible Rate(BPR)...', ncols=100, leave=False) if rank == 0 else train_loader
        PBR_params = [input_size, criterion.num_anchor_per_scale, criterion.anchors, criterion.strides]
        total_n_train, total_n_target = check_best_possible_recall(dataloader, PBR_params, config_item['ANCHOR_IOU_THRESHOLD'])
        
        if OS_SYSTEM == 'Linux':
            total_n_train = torch.tensor(total_n_train).cuda(rank)
            total_n_target = torch.tensor(total_n_target).cuda(rank)
            dist.all_reduce(total_n_train, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_n_target, op=dist.ReduceOp.SUM)

        if rank == 0:
            message = f'Best Possible Rate: {total_n_train/total_n_target:0.4f}, Train_target/Total_target: {total_n_train}/{total_n_target}\n'
            logging.warning(message)
        del dataloader

    #################################### Init Model ####################################
    torch.manual_seed(2023)
    torch.cuda.set_device(rank)
    model = model.cuda(rank)

    if OS_SYSTEM == 'Linux':
        model = DDP(model, device_ids=[rank])
    
    #################################### Load Model ####################################
    if OS_SYSTEM == 'Linux':
        dist.barrier() # let all processes sync up before starting
    
    if config_item['WEIGHT_PATH'] is not None:
        if rank == 0:
            logging.warning(f'Path to pretrained model: {config_item["WEIGHT_PATH"]}')
        map_location = {'cpu':'cuda:%d' %rank}
        ckpt = torch.load(config_item['WEIGHT_PATH'], map_location=map_location)
        if hasattr(model, 'module'):
            model.module.load_state_dict(ckpt, strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

    #################################### Train Model ####################################
    best_mAP = args.init_score
    progress_bar = tqdm(range(1, config_item['NUM_EPOCHS']+1), ncols=100) if rank == 0 else range(1, config_item['NUM_EPOCHS']+1)

    for epoch in progress_bar:
        if rank == 0:
            message = f'[Epoch:{epoch:03d}/{len(progress_bar):03d}]'
            progress_bar.set_description(desc=message)
            train_loader = tqdm(train_loader, desc='[Phase:TRAIN]', ncols=100, leave=False)
            val_loader = tqdm(val_loader, desc='[Phase:VAL]', ncols=100, leave=False)
            
        train_loss = execute_train(rank=rank, dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer)
        val_loss, gather_objects, canvas = execute_val(rank=rank, world_size=world_size, 
                                                        config=config_item, dataloader=val_loader,
                                                        model=model, criterion=criterion, evaluator=evaluator, 
                                                        class_list=class_list, color_list=color_list)
        scheduler.step()

        if rank == 0:
            monitor_text = f' Train Loss: {train_loss["total"]/world_size:.2f}, Val Loss: {val_loss["total"]/world_size:.2f}'
            detections = []
            for det in gather_objects:
                detections.extend(det)
            mAP_info, eval_text = evaluator(detections)
            logging.warning(message + monitor_text)
            logging.warning(eval_text)

            if mAP_info['all']['mAP05'] > best_mAP:
                best_mAP = mAP_info['all']['mAP05']
                model_to_save = model.module if hasattr(model, 'module') else model
                save_model(model=deepcopy(model_to_save).cpu(), 
                            save_path=args.exp_path / 'weights', 
                            model_name=f'{TIMESTAMP}-EP{epoch:02d}.pth')                    

            if epoch % args.img_log_interval == 0:
                imwrite(str(args.exp_path / 'images' / f'{TIMESTAMP}-EP{epoch:02d}.jpg'), canvas)
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/coco128.yml', help='Path to data.yml file')
    parser.add_argument('--config_path', type=str, default='config/yolov3.yml', help='Path to config.yml file')
    parser.add_argument('--exp_name', type=str, default=str(TIMESTAMP), help='Name to log training')
    parser.add_argument('--gpu_ids', type=int, help='List of GPU IDs', default=[0], nargs='+')
    parser.add_argument('--img_log_interval', type=int, default=1, help='Image logging interval')
    parser.add_argument('--init_score', type=float, default=0.1, help='initial mAP score for update best model')
    args = parser.parse_args()
    args.data_path = ROOT / args.data_path
    args.config_path = ROOT / args.config_path
    args.exp_path = ROOT / 'experiments' / args.exp_name
    os.makedirs(args.exp_path / 'images', exist_ok=True)
    os.makedirs(args.exp_path / 'logs', exist_ok=True)
    world_size = len(args.gpu_ids)
    assert world_size > 0, 'Executable GPU machine does not exist, This training supports on CUDA available environment.'

    #########################################################
    # Set multiprocessing type to spawn
    torch.multiprocessing.set_start_method('spawn', force=True)
    logger = setup_primary_logging(args.exp_path / 'logs' / f'{TIMESTAMP}.log')
    mp.spawn(main_work, args=(world_size, args, logger), nprocs=world_size, join=True)
    #########################################################

if __name__ == '__main__':
    main()