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
            prediction = predictions[idx].cpu().numpy()
            prediction[:, :4] = box_transform_xcycwh_to_x1y1x2y2(prediction[:, :4], clip_max=config['INPUT_SIZE'])
            prediction = filter_obj_score(prediction=prediction, conf_threshold=config['MIN_SCORE_THRESH'])
            prediction = run_NMS(prediction, iou_threshold=config['MIN_IOU_THRESH'], maxDets=config['MAX_DETS'], class_agnostic=False)

            if len(prediction) > 0:
                prediction[:, 1:5] /= config['INPUT_SIZE']
                detections.append((filename, prediction, max_size))

    _, eval_text = evaluator(detections)
    return eval_text


def main_work(args):
    with open(args.data, mode='r') as f:
        data_item = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.config, mode='r') as f:
        config_item = yaml.load(f, Loader=yaml.FullLoader)

    input_size = config_item['INPUT_SIZE']
    batch_size = config_item['BATCH_SIZE']
    max_dets = config_item['MAX_DETS']
    anchors = list(config_item['ANCHORS'].values())
    class_list = data_item['NAMES']
    val_set = Dataset(args=args, phase='val', rank=args.rank, time_created=TIMESTAMP)
    val_loader = DataLoader(dataset=val_set, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True)
    model = YOLOv3_Model(input_size=input_size, anchors=anchors, num_classes=len(class_list))
    val_file = args.data.parent / data_item['mAP_FILE']
    assert val_file.is_file(), RuntimeError(f'Not exist val file, expected {val_file}')
    evaluator = Evaluator(GT_file=val_file, maxDets=max_dets)
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.cuda(args.rank)
    eval_text = execute_val(rank=args.rank, config=config_item, dataloader=val_loader, model=model, evaluator=evaluator)
    print(eval_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='path to data.yaml file')
    parser.add_argument('--config', type=str, default='config/yolov3_coco.yaml', help='path to config.yaml file')
    parser.add_argument('--model', type=str, default='weights/voc-b64/model/best.pt', help='path to trained model weight')
    parser.add_argument('--rank', type=int, default=0, help='GPU device index for running')
    args = parser.parse_args()
    args.data = ROOT / args.data
    args.config = ROOT / args.config
    args.model = ROOT / args.model
    torch.cuda.set_device(args.rank)
    main_work(args)


if __name__ == '__main__':
    main()
