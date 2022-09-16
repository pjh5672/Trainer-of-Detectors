import os
import sys
import time
import math
import shutil
import logging
import platform
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timedelta

import yaml
import torch
import numpy as np
from torch import nn
from torch.cuda import amp
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from thop import profile
from tqdm import tqdm, trange

from dataloader import Dataset, to_image, denormalize
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


def execute_train(rank, args, dataloader, model, criterion, optimizer, scaler, class_list, color_list):
    global accumulate, last_opt_step
    loss_types = ['total', 'coord', 'obj', 'noobj', 'cls']
    total_loss = 0.0
    model.train()
    optimizer.zero_grad()

    for index, mini_batch in enumerate(dataloader):
        if index == 0:
            canvas_img = to_image(denormalize(mini_batch[0][0]))
            canvas_gt = mini_batch[1][0]

        ni = index + len(dataloader) * current_epoch
        if ni <= nw:
            xi = [0, nw]
            accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * lf(current_epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])

        images, targets = mini_batch[0].cuda(rank, non_blocking=True), mini_batch[1]
        with amp.autocast(enabled=not args.no_amp):
            predictions = model(images)
            losses = criterion(predictions, targets)

        scaler.scale(losses[0]).backward()
        if ni - last_opt_step >= accumulate:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            last_opt_step = ni

        monitor_text = ''
        for loss_name, loss_value in zip(loss_types, losses):
            if not torch.isfinite(loss_value) and loss_name != 'total':
                print(f'############## {loss_name} Loss is Nan/Inf ! {loss_value} ##############')
                sys.exit(0)
            if loss_name == 'total':
                total_loss += loss_value
            else:
                monitor_text += f'{loss_name}: {loss_value.item():.3f} '
        if rank == 0:
            dataloader.set_postfix_str(s=f'{monitor_text}')

    total_loss /= len(dataloader)
    if OS_SYSTEM == 'Linux':
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

    if (current_epoch >= args.start_eval) and (rank == 0):
        canvas = visualize_target(canvas_img, canvas_gt, class_list, color_list)
    else:
        canvas = None
    del images, predictions, losses
    torch.cuda.empty_cache()
    return total_loss, canvas


@torch.no_grad()
def execute_val(rank, world_size, args, config, dataloader, model, criterion, class_list, color_list):
    loss_types = ['total', 'coord', 'obj', 'noobj', 'cls']
    total_loss = 0.0
    gather_objects = [None, ] * world_size
    detections = []
    model.eval()

    for index, mini_batch in enumerate(dataloader):
        if index == 0:
            canvas_img = to_image(denormalize(mini_batch[0][0]))

        images, targets, filenames, max_sizes = mini_batch[0].cuda(rank, non_blocking=True), mini_batch[1], mini_batch[2], mini_batch[3]
        predictions = model(images)
        losses = criterion(predictions, targets)
        predictions = torch.cat(predictions, dim=1)
        predictions[..., 4:] = torch.sigmoid(predictions[..., 4:])
        predictions[..., 5:] *= predictions[..., 4:5]

        if current_epoch >= args.start_eval:
            for idx in range(len(filenames)):
                filename = filenames[idx]
                max_size = max_sizes[idx]
                prediction = predictions[idx].cpu().numpy()
                prediction[:, :4] = box_transform_xcycwh_to_x1y1x2y2(prediction[:, :4], clip_max=config['INPUT_SIZE'])
                prediction = filter_obj_score(prediction=prediction, conf_threshold=config['MIN_SCORE_THRESH'])
                prediction = run_NMS(prediction, iou_threshold=config['MIN_IOU_THRESH'], maxDets=config['MAX_DETS'], class_agnostic=False)

                if len(prediction) > 0:
                    prediction[:, 1:5] /= config['INPUT_SIZE']
                    detections.append((filename, prediction, max_size))

        monitor_text = ''
        for loss_name, loss_value in zip(loss_types, losses):
            if not torch.isfinite(loss_value) and loss_name != 'total':
                print(f'############## {loss_name} Loss is Nan/Inf ! {loss_value} ##############')
                sys.exit(0)
            if loss_name == 'total':
                total_loss += loss_value
            else:
                monitor_text += f'{loss_name}: {loss_value.item():.3f} '
        if rank == 0:
            dataloader.set_postfix_str(s=f'{monitor_text}')

    total_loss /= len(dataloader)
    if OS_SYSTEM == 'Linux':
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_gather_object(gather_objects, detections)
    elif OS_SYSTEM == 'Windows':
        gather_objects = [detections]

    if (current_epoch >= args.start_eval) and (rank == 0):
        canvas = visualize_prediction(canvas_img, detections[0][1], 0.1, class_list, color_list)
    else:
        canvas = None
    del images, predictions, losses
    torch.cuda.empty_cache()
    return total_loss, gather_objects, canvas


def main_work(rank, world_size, args, logger):
    ################################### Init Params ###################################
    global current_epoch, nbs, nw, lf, momentum, accumulate, batch_size, warmup_bias_lr, warmup_momentum, last_opt_step

    if OS_SYSTEM == 'Linux':
        setup_worker_logging(rank, logger)
    shutil.copy(args.data, args.exp_path / args.data.name)
    shutil.copy(args.config, args.exp_path / args.config.name)

    with open(args.data, mode='r') as f:
        data_item = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.config, mode='r') as f:
        config_item = yaml.load(f, Loader=yaml.FullLoader)

    input_size = config_item['INPUT_SIZE']
    input_channel = config_item['INPUT_CHANNEL']
    lr0 = config_item['INIT_LEARNING_RATE']
    lrf = config_item['FINAL_LEARNING_RATE']
    weight_decay = config_item['WEIGHT_DECAY']
    num_epochs = config_item['NUM_EPOCHS']
    momentum = config_item['MOMENTUM']
    batch_size = config_item['BATCH_SIZE']
    warmup_epoch = config_item['WARMUP_EPOCH']
    warmup_momentum = config_item['WARMUP_MOMENTUM']
    warmup_bias_lr = config_item['WARMUP_BIAS_LR']
    class_list = data_item['NAMES']
    color_list = generate_random_color(num_colors=len(class_list))
    train_set = Dataset(args=args, phase='train', rank=rank, time_created=TIMESTAMP)
    val_set = Dataset(args=args, phase='val', rank=rank, time_created=TIMESTAMP)
    model = YOLOv3_Model(config_path=args.config, num_classes=len(class_list), freeze_backbone=args.freeze_backbone)
    criterion = YOLOv3_Loss(config_path=args.config, model=model)

    if rank == 0:
        logging.warning(f'{train_set.data_info}')
        logging.warning(f'{val_set.data_info}')
        macs, params = profile(deepcopy(model), inputs=(torch.randn(1, input_channel, input_size, input_size),), verbose=False)
        logging.warning(f'Params(M): {params/1e+6:.2f}, FLOPS(B): {2*macs/1E+9:.2f}')

    nbs = 64 # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    weight_decay *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = build_optimizer(args=args, model=model, lr=lr0, momentum=momentum, weight_decay=weight_decay)

    val_file = args.data.parent / data_item['mAP_FILE']
    assert val_file.is_file(), RuntimeError(f'Not exist val file, expected {val_file}')
    evaluator = Evaluator(GT_file=val_file, maxDets=config_item['MAX_DETS'])

    if args.linear_lr:
        lf = lambda x: (1 - x / num_epochs) * (1.0 - lrf) + lrf
    else:
        lf = lambda x: ((1 - math.cos(x * math.pi / num_epochs)) / 2) * (lrf - 1) + 1
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = amp.GradScaler(enabled=not args.no_amp)

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
        dataloader = tqdm(train_loader, desc='Calculating Best Possible Rate(BPR)...', ncols=115, leave=False) if rank == 0 else train_loader
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
            message = f'Best Possible Rate: {total_n_train/total_n_target:.4f}, Train_target/Total_target: {total_n_train}/{total_n_target}'
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

    if config_item['RESUME_PATH'] is not None:
        if rank == 0:
            logging.warning(f'Path to resume model: {config_item["RESUME_PATH"]}\n')
        checkpoint = torch.load(config_item['RESUME_PATH'], map_location='cpu')
        start_epoch = checkpoint['epoch']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(rank)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        start_epoch = 0
        map_location = {'cpu':'cuda:%d' %rank}
        if config_item['PRETRAINED_PATH'] is not None:
            if rank == 0:
                logging.warning(f'Path to pretrained model: {config_item["PRETRAINED_PATH"]}\n')
            checkpoint = torch.load(config_item['PRETRAINED_PATH'], map_location=map_location)
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

    #################################### Train Model ####################################
    best_mAP = 0.1
    best_perf = None
    last_opt_step = -1
    nw = max(round(warmup_epoch * len(train_loader)), 100)
    pbar = trange(start_epoch, num_epochs, total=num_epochs, initial=start_epoch, ncols=115) if rank == 0 else range(start_epoch, num_epochs)

    for i in pbar:
        current_epoch = i

        if rank == 0:
            message = f'[Epoch:{current_epoch:03d}/{num_epochs:03d}]'
            pbar.set_description(desc=message)
            train_loader = tqdm(train_loader, desc='[Phase:TRAIN]', ncols=115, leave=False)
            val_loader = tqdm(val_loader, desc='[Phase:VAL]', ncols=115, leave=False)

        train_sampler.set_epoch(current_epoch)
        train_loss, canvas_train = execute_train(rank=rank, args=args, dataloader=train_loader, model=model,
                                                 criterion=criterion, optimizer=optimizer, scaler=scaler,
                                                 class_list=class_list, color_list=color_list)
        val_loss, gather_objects, canvas_val = execute_val(rank=rank, args=args, world_size=world_size, config=config_item,
                                                           dataloader=val_loader, model=model, criterion=criterion,
                                                           class_list=class_list, color_list=color_list)
        if current_epoch >= warmup_epoch:
            scheduler.step()

        if rank == 0:
            monitor_text = f' Train Loss: {train_loss/world_size:.2f}, Val Loss: {val_loss/world_size:.2f}'
            logging.warning(message + monitor_text)

            if current_epoch % args.img_interval == 0:
                if canvas_train is not None:
                    imwrite(str(args.image_log_dir / 'train' / f'EP{current_epoch:03d}.jpg'), canvas_train)
                if canvas_val is not None:
                    imwrite(str(args.image_log_dir / 'val' / f'EP{current_epoch:03d}.jpg'), canvas_val)

            if current_epoch >= args.start_eval:
                start = time.time()
                detections = []
                for det in gather_objects:
                    detections.extend(det)
                mAP_info, eval_text = evaluator(detections)
                logging.warning(message + f' mAP Computation Time(sec): {time.time() - start:.4f}')
                logging.warning(eval_text)

                model_to_save = deepcopy(model.module).cpu() if hasattr(model, 'module') else deepcopy(model).cpu()
                save_item = {'epoch': current_epoch,
                             'class_list': class_list,
                             'model_state_dict': model_to_save.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'scaler_state_dict': scaler.state_dict()}
                save_model(model=save_item, save_path=args.weight_dir / 'last.pt')

                if mAP_info['all']['mAP_50'] > best_mAP:
                    best_mAP = mAP_info['all']['mAP_50']
                    best_epoch = current_epoch
                    best_perf = eval_text
                    save_model(model=save_item, save_path=args.weight_dir / 'best.pt')

                    analysis_result = analyse_mAP_info(mAP_info['all'], class_list)
                    data_df, figure_AP, figure_dets, fig_PR_curves = analysis_result
                    data_df.to_csv(str(args.analysis_log_dir / f'dataframe_EP{current_epoch:03d}.csv'))
                    figure_AP.savefig(str(args.analysis_log_dir / f'figure-AP_EP{current_epoch:03d}.png'))
                    figure_dets.savefig(str(args.analysis_log_dir / f'figure-dets_EP{current_epoch:03d}.png'))
                    PR_curve_dir = args.analysis_log_dir / 'PR_curve' / f'EP{current_epoch:03d}'
                    os.makedirs(PR_curve_dir, exist_ok=True)
                    for class_id in fig_PR_curves.keys():
                        fig_PR_curves[class_id].savefig(str(PR_curve_dir / f'{class_list[class_id]}.png'))
                        fig_PR_curves[class_id].clf()

    if (rank == 0) and (best_perf is not None):
        logging.warning(f' Best mAP@0.5: {best_mAP:.3f} at [Epoch:{best_epoch}/{num_epochs}]')
        logging.warning(best_perf)

    cleanup()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='path to data.yaml file')
    parser.add_argument('--config', type=str, default='config/yolov3_coco.yaml', help='path to config.yaml file')
    parser.add_argument('--exp_name', type=str, default=str(TIMESTAMP), help='name to log training')
    parser.add_argument('--world_size', type=int, default=1, help='number of available GPU devices')
    parser.add_argument('--img_interval', type=int, default=10, help='image logging interval')
    parser.add_argument('--adam', action='store_true', help='use of Adam optimizer (default: SGD optimizer)')
    parser.add_argument('--linear_lr', action='store_true', help='use of linear LR scheduler (default: one cyclic scheduler)')
    parser.add_argument('--no_amp', action='store_true', help='use of FP32 training (default: AMP training)')
    parser.add_argument('--start_eval', type=int, default=50, help='starting epoch for mAP evaluation')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze backbone layers (default: False)')

    args = parser.parse_args()
    args.data = ROOT / args.data
    args.config = ROOT / args.config
    args.exp_path = ROOT / 'experiments' / args.exp_name
    args.weight_dir = args.exp_path / 'models'
    args.image_log_dir = args.exp_path / 'images'
    args.analysis_log_dir = args.exp_path / 'analysis'
    assert args.world_size > 0, 'Executable GPU machine does not exist, This training supports on CUDA available environment.'

    os.makedirs(args.weight_dir, exist_ok=True)
    os.makedirs(args.image_log_dir / 'train', exist_ok=True)
    os.makedirs(args.image_log_dir / 'val', exist_ok=True)
    os.makedirs(args.analysis_log_dir, exist_ok=True)

    with open(args.exp_path / 'args.yaml', mode='w') as f:
        args_dict = {}
        for k, v in vars(args).items():
            args_dict[k] = str(v) if isinstance(v, Path) else v
        yaml.safe_dump(args_dict, f, sort_keys=False)

    #########################################################
    # Set multiprocessing type to spawn
    if OS_SYSTEM == 'Linux':
        torch.multiprocessing.set_start_method('spawn', force=True)
        logger = setup_primary_logging(args.exp_path / 'train.log')
        mp.spawn(main_work, args=(args.world_size, args, logger), nprocs=args.world_size, join=True)
    elif OS_SYSTEM == 'Windows':
        logger = build_win_logger(args.exp_path / 'train.log')
        main_work(rank=0, world_size=1, args=args, logger=logger)
    #########################################################

if __name__ == '__main__':
    main()
