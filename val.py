import argparse
from pathlib import Path
from datetime import datetime

import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import Dataset
from models import YOLOv3_Model
from utils import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')
seed_num = 2023



@torch.no_grad()
def execute_val(rank, config, dataloader, model, evaluator):
    model.eval()
    detections = []

    for _, mini_batch in enumerate(tqdm(dataloader, desc='[Phase:VAL]', ncols=115)):
        images, filenames, max_sizes = mini_batch[0].cuda(rank), mini_batch[2], mini_batch[3]
        predictions = model(images)
        predictions = torch.cat(predictions, dim=1)
        predictions[..., 4:] = torch.sigmoid(predictions[..., 4:])
        predictions[..., 5:] *= predictions[..., 4:5]

        for idx in range(len(filenames)):
            filename = filenames[idx]
            max_size = max_sizes[idx]
            pred_yolo = predictions[idx].cpu().numpy()
            pred_yolo[:, :4] = clip_box_coordinates(bboxes=pred_yolo[:, :4]/config['INPUT_SIZE'])
            pred_yolo = filter_obj_score(prediction=pred_yolo, conf_threshold=config['MIN_SCORE_THRESH'])
            pred_yolo = run_NMS_for_YOLO(prediction=pred_yolo, iou_threshold=config['MIN_IOU_THRESH'], maxDets=config['MAX_DETS'])
            if len(pred_yolo) > 0:
                detections.append((filename, pred_yolo, max_size))

    _, eval_text = evaluator(detections)
    return eval_text


def main_work(args):
    with open(args.data_path, mode='r') as f:
        data_item = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.config_path, mode='r') as f:
        config_item = yaml.load(f, Loader=yaml.FullLoader)

    batch_size = config_item['BATCH_SIZE']
    max_dets = config_item['MAX_DETS']
    class_list = data_item['NAMES']

    val_set = Dataset(args=args, phase='val', rank=args.rank, time_created=TIMESTAMP)
    val_loader = DataLoader(dataset=val_set, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True)
    model = YOLOv3_Model(config_path=args.config_path, num_classes=len(class_list))
    val_file = args.data_path.parent / data_item['mAP_FILE']
    assert val_file.is_file(), RuntimeError(f'Not exist val file, expected {val_file}')
    evaluator = Evaluator(GT_file=val_file, maxDets=max_dets)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.cuda(args.rank)

    eval_text = execute_val(rank=args.rank, config=config_item, dataloader=val_loader, model=model, evaluator=evaluator)
    print(eval_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/coco128.yaml', help='path to data.yaml file')
    parser.add_argument('--config_path', type=str, default='config/yolov3_coco.yaml', help='path to config.yaml file')
    parser.add_argument('--model_path', type=str, default='weights/voc_best.pt', help='path to trained model weight')
    parser.add_argument('--rank', type=int, default=0, help='GPU device index for running')
    args = parser.parse_args()
    args.data_path = ROOT / args.data_path
    args.config_path = ROOT / args.config_path
    args.model_path = ROOT / args.model_path
    torch.cuda.set_device(args.rank)
    main_work(args)


if __name__ == '__main__':
    main()
