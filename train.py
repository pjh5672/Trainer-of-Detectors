import os
import time
import shutil
import logging
import platform
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta

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
seed_num = 2023

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


def execute_train(rank, dataloader, model, criterion, optimizer, class_list, color_list):
    loss_types = ['total', 'coord', 'obj', 'noobj', 'cls']
    total_loss = 0.0
    model.train()

    for index, mini_batch in enumerate(dataloader):
        if index == 0:
            canvas_img = mini_batch[0][0]
            canvas_gt = mini_batch[1][0]

        images = mini_batch[0].cuda(rank, non_blocking=True)
        targets = mini_batch[1]

        predictions = model(images)
        losses = criterion(predictions, targets)
        optimizer.zero_grad()
        losses[0].backward()
        optimizer.step()

        monitor_text = ''
        for loss_name, loss_value in zip(loss_types, losses):
            if loss_name == 'total':
                total_loss += loss_value
            else:
                monitor_text += f'{loss_name}: {loss_value.item():.2f} '
        if rank == 0:
            dataloader.set_postfix_str(s=f'{monitor_text}')

    total_loss /= len(dataloader)

    if OS_SYSTEM == 'Linux':
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

    canvas = visualize_target(canvas_img, canvas_gt, class_list, color_list) if rank == 0 else None
    del images, predictions, losses
    torch.cuda.empty_cache()
    return total_loss, canvas


@torch.no_grad()
def execute_val(rank, world_size, config, dataloader, model, criterion, class_list, color_list):
    loss_types = ['total', 'coord', 'obj', 'noobj', 'cls']
    total_loss = 0.0
    gather_objects = [None, ] * world_size
    detections = []
    model.eval()

    for index, mini_batch in enumerate(dataloader):
        if index == 0:
            canvas_img = mini_batch[0][0]

        images = mini_batch[0].cuda(rank, non_blocking=True)
        targets = mini_batch[1]
        filenames = mini_batch[2]
        max_sides = mini_batch[3]

        predictions = model(images)
        losses = criterion(predictions, targets)
        predictions = torch.cat(predictions, dim=1)
        predictions[..., 4:] = torch.sigmoid(predictions[..., 4:])
        predictions[..., 5:] *= predictions[..., 4:5]

        for idx in range(len(filenames)):
            filename = filenames[idx]
            max_side = max_sides[idx]
            pred_yolo = predictions[idx].cpu().numpy()
            pred_yolo[:, :4] = clip_box_coordinates(bboxes=pred_yolo[:, :4]/config['INPUT_SIZE'])
            pred_yolo = filter_obj_score(prediction=pred_yolo, conf_threshold=config['MIN_SCORE_THRESH'])
            pred_yolo = run_NMS_for_YOLO(prediction=pred_yolo, iou_threshold=config['MIN_IOU_THRESH'], maxDets=config['MAX_DETS'])
            if len(pred_yolo) > 0:
                detections.append((filename, pred_yolo, max_side))

        monitor_text = ''
        for loss_name, loss_value in zip(loss_types, losses):
            if loss_name == 'total':
                total_loss += loss_value
            else:
                monitor_text += f'{loss_name}: {loss_value.item():.2f} '
        if rank == 0:
            dataloader.set_postfix_str(s=f'{monitor_text}')

    total_loss /= len(dataloader)

    if OS_SYSTEM == 'Linux':
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_gather_object(gather_objects, detections)
    elif OS_SYSTEM == 'Windows':
        gather_objects = [detections]

    min_conf_thresh = 0.1 # just for checking val result
    canvas = visualize_prediction(canvas_img, detections[0][1], min_conf_thresh, class_list, color_list) if rank == 0 else None
    del images, predictions, losses
    torch.cuda.empty_cache()
    return total_loss, gather_objects, canvas


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
    init_lr = config_item['INIT_LEARNING_RATE']
    weight_decay = config_item['WEIGHT_DECAY']
    num_epochs = config_item['NUM_EPOCHS']
    augment_strong = config_item['AUGMENT_STRONG']
    batch_size = config_item['BATCH_SIZE']
    lrf = config_item['LEARNING_RATE_SCALE']
    color_list = generate_random_color(num_colors=len(class_list))
    transformer = build_transformer(input_size=(input_size, input_size), augment_strong=augment_strong)
    train_set = Dataset(data_path=args.data_path, phase='train', rank=rank, time_created=TIMESTAMP, transformer=transformer['train'])
    val_set = Dataset(data_path=args.data_path, phase='val', rank=rank, time_created=TIMESTAMP, transformer=transformer['val'])

    if rank == 0:
        logging.warning(f'{train_set.data_info}')
        logging.warning(f'{val_set.data_info}')

    model = YOLOv3_Model(config_path=args.config_path, num_classes=len(class_list))
    criterion = YOLOv3_Loss(config_path=args.config_path, model=model)
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=weight_decay)
    val_file = args.data_path.parent / data_item['mAP_FILE']
    assert val_file.is_file(), RuntimeError(f'Not exist val file, expected {val_file}')
    evaluator = Evaluator(GT_file=val_file, config=config_item)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 - 10 ** (lrf * (1 - x / num_epochs)))

    ################################### Init Process ###################################
    setup(rank, world_size)

    ################################### Init Loader ####################################
    batch_size = int(batch_size/world_size)
    num_workers = min([os.cpu_count() // max(torch.cuda.device_count(), 1), batch_size if batch_size > 1 else 0, world_size*4])
    train_sampler = distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = distributed.DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(dataset=train_set, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_set, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, sampler=val_sampler)

    ################################### Calculate BPR ####################################
    if config_item['GET_PBR']:
        dataloader = tqdm(train_loader, desc='Calculating Best Possible Rate(BPR)...', ncols=110, leave=False) if rank == 0 else train_loader
        num_anchor_per_scale = criterion.num_anchor_per_scale
        anchors = criterion.anchors
        strides = criterion.strides
        anchor_iou_threshold = config_item['ANCHOR_IOU_THRESHOLD']
        total_n_train, total_n_target = check_best_possible_recall(dataloader, input_size, num_anchor_per_scale, anchors, strides, anchor_iou_threshold)

        if OS_SYSTEM == 'Linux':
            total_n_train = torch.tensor(total_n_train).cuda(rank)
            total_n_target = torch.tensor(total_n_target).cuda(rank)
            dist.all_reduce(total_n_train, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_n_target, op=dist.ReduceOp.SUM)

        if rank == 0:
            message = f'Best Possible Rate: {total_n_train/total_n_target:.4f}, Train_target/Total_target: {total_n_train}/{total_n_target}\n'
            logging.warning(message)

        del dataloader

    #################################### Init Model ####################################
    torch.manual_seed(seed_num)
    torch.cuda.set_device(rank)
    model = model.cuda(rank)

    if OS_SYSTEM == 'Linux':
        model = DDP(model, device_ids=[rank])

    #################################### Load Model ####################################
    if OS_SYSTEM == 'Linux':
        dist.barrier() # let all processes sync up before starting

    if config_item['WEIGHT_PATH'] is not None:
        if rank == 0:
            logging.warning(f'Path to pretrained model: {config_item["WEIGHT_PATH"]}\n')
        map_location = {'cpu':'cuda:%d' %rank}
        ckpt = torch.load(config_item['WEIGHT_PATH'], map_location=map_location)
        if hasattr(model, 'module'):
            model.module.load_state_dict(ckpt, strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

    #################################### Train Model ####################################
    best_mAP = args.init_score
    progress_bar = tqdm(range(1, num_epochs+1), ncols=110) if rank == 0 else range(1, num_epochs+1)

    for epoch in progress_bar:
        if rank == 0:
            message = f'[Epoch:{epoch:03d}/{len(progress_bar):03d}]'
            progress_bar.set_description(desc=message)
            train_loader = tqdm(train_loader, desc='[Phase:TRAIN]', ncols=110, leave=False)
            val_loader = tqdm(val_loader, desc='[Phase:VAL]', ncols=110, leave=False)

        train_sampler.set_epoch(epoch)
        train_loss, canvas_train = execute_train(rank=rank, dataloader=train_loader, model=model,
                                                 criterion=criterion, optimizer=optimizer,
                                                 class_list=class_list, color_list=color_list)
        val_loss, gather_objects, canvas_val = execute_val(rank=rank, world_size=world_size, config=config_item,
                                                           dataloader=val_loader, model=model, criterion=criterion,
                                                           class_list=class_list, color_list=color_list)
        scheduler.step()

        if rank == 0:
            monitor_text = f' Train Loss: {train_loss/world_size:.2f}, Val Loss: {val_loss/world_size:.2f}'
            logging.warning(message + monitor_text)

            start = time.time()
            detections = []
            for det in gather_objects:
                detections.extend(det)
            mAP_info, eval_text = evaluator(detections)
            logging.warning(message + f' mAP Computation Time(sec): {time.time() - start:.4f}')
            logging.warning(eval_text)

            if epoch % args.img_interval == 0:
                imwrite(str(args.image_log_dir / 'train' / f'EP{epoch:03d}.jpg'), canvas_train)
                imwrite(str(args.image_log_dir / 'val' / f'EP{epoch:03d}.jpg'), canvas_val)

            if epoch >= args.start_save:
                if mAP_info['all']['mAP_50'] > best_mAP:
                    best_mAP = mAP_info['all']['mAP_50']
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.class_list = class_list
                    save_model(model=deepcopy(model_to_save).cpu(), save_path=args.weight_dir / f'EP{epoch:03d}.pt')

                    analysis_result = analyse_mAP_info(mAP_info['all'], class_list)
                    data_df, figure_AP, figure_dets, fig_PR_curves = analysis_result
                    data_df.to_csv(str(args.analysis_log_dir / f'dataframe_EP{epoch:03d}.csv'))
                    figure_AP.savefig(str(args.analysis_log_dir / f'figure-AP_EP{epoch:03d}.png'))
                    figure_dets.savefig(str(args.analysis_log_dir / f'figure-dets_EP{epoch:03d}.png'))

                    PR_curve_dir = args.analysis_log_dir / 'PR_curve' / f'EP{epoch:03d}'
                    os.makedirs(PR_curve_dir, exist_ok=True)
                    for class_id in fig_PR_curves.keys():
                        fig_PR_curves[class_id].savefig(str(PR_curve_dir / f'{class_list[class_id]}.png'))
    if rank == 0:
        logging.warning(f' Best mAP@0.5: {best_mAP:.3f}')
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/coco128.yml', help='Path to data.yml file')
    parser.add_argument('--config_path', type=str, default='config/yolov3.yml', help='Path to config.yml file')
    parser.add_argument('--exp_name', type=str, default=str(TIMESTAMP), help='Name to log training')
    parser.add_argument('--gpu_ids', type=int, default=[0], nargs='+', help='List of GPU IDs')
    parser.add_argument('--img_interval', type=int, default=5, help='Image logging interval')
    parser.add_argument('--start_save', type=int, default=10, help='Starting model saving epoch')
    parser.add_argument('--init_score', type=float, default=0.1, help='Initial mAP score for update best model')

    args = parser.parse_args()
    args.data_path = ROOT / args.data_path
    args.config_path = ROOT / args.config_path
    args.exp_path = ROOT / 'experiments' / args.exp_name
    args.weight_dir = args.exp_path / 'weights'
    args.image_log_dir = args.exp_path / 'images'
    args.analysis_log_dir = args.exp_path / 'analysis'

    os.makedirs(args.weight_dir, exist_ok=True)
    os.makedirs(args.image_log_dir / 'train', exist_ok=True)
    os.makedirs(args.image_log_dir / 'val', exist_ok=True)
    os.makedirs(args.analysis_log_dir, exist_ok=True)
    world_size = len(args.gpu_ids)
    assert world_size > 0, 'Executable GPU machine does not exist, This training supports on CUDA available environment.'

    #########################################################
    # Set multiprocessing type to spawn
    torch.multiprocessing.set_start_method('spawn', force=True)
    logger = setup_primary_logging(args.exp_path / 'train.log')
    mp.spawn(main_work, args=(world_size, args, logger), nprocs=world_size, join=True)
    #########################################################

if __name__ == '__main__':
    main()
