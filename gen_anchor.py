import argparse
from pathlib import Path
from datetime import datetime

import yaml
from tqdm import tqdm
from scipy.cluster.vq import kmeans

from dataloader import Dataset
from utils import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')



def generate_best_anchors(args, ratio_thres=4, min_size=4, n_generation=1000):
    ratio_thres = 1 / ratio_thres
    train_set = Dataset(args=args, phase='train', rank=0, time_created=TIMESTAMP)
    pbar = tqdm(train_set, desc='Collecting Annotation info...', ncols=115, leave=False)

    img_wh = []
    anno_bbox_wh = []
    anno_cls_ids = []
    for index, _ in enumerate(pbar):
        image, target = train_set.get_item(index)[1:]
        img_wh.append([image.shape[1], image.shape[0]])
        anno_bbox_wh.append(target[:, -2:].tolist())
        anno_cls_ids.append(target[:, 0].tolist())

    img_wh = np.array(img_wh, dtype=np.float32)
    shapes = img_wh / img_wh.max(axis=1, keepdims=True) * args.img_size
    bbox_wh = np.concatenate([l * s for s, l in zip(shapes, anno_bbox_wh)])
    bbox_wh_2 = bbox_wh[(bbox_wh >= min_size).any(axis=1)].astype(np.float32)
    s = bbox_wh_2.std(axis=0)
    k = kmeans(obs=bbox_wh_2/s, k_or_guess=args.n_cluster, iter=500)[0] * s

    bbox_wh_2, bbox_wh = (torch.tensor(x, dtype=torch.float32) for x in (bbox_wh_2, bbox_wh))
    k, bpr = get_anchors(k, bbox_wh, ratio_thres)
    f, sh, mp, s = anchor_fitness(k, bbox_wh_2, ratio_thres), k.shape, 0.9, 0.1

    pbar = tqdm(range(n_generation), ncols=115, leave=False)
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((np.random.random(sh) < mp) * random.random() * np.random.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=min_size)
        fg = anchor_fitness(kg, bbox_wh_2, ratio_thres)
        if fg > f:
            f, k = fg, kg.copy()
            fg = anchor_fitness(k, bbox_wh_2, ratio_thres)
            message = f'Evolving anchors with Genetic Algorithm - fitness:{f:.4f}, BPR:{bpr:.4f}'
            pbar.set_description(desc=message)
        k, bpr = get_anchors(k, bbox_wh, ratio_thres)
    k = [anchor.round(0).tolist() for anchor in k]
    return anno_cls_ids, k, bpr.item()


def metric(k, wh):
    r = wh[:, None] / k[None]
    x = torch.min(r, 1 / r).min(2)[0]
    return x.max(1)[0]


def anchor_fitness(k, wh, ratio_thres):
    best = metric(torch.tensor(k, dtype=torch.float32), wh)
    return (best * (best > ratio_thres).float()).mean() # mutation fitness


def get_anchors(k, wh, ratio_thres):
    k = k[np.argsort(k.prod(axis=1))]
    best = metric(k, wh)
    bpr = (best > ratio_thres).float().mean()
    return k, bpr


def main_work(args):
    with open(args.data, mode='r') as f:
        data_item = yaml.load(f, Loader=yaml.FullLoader)
    cls_ids, anchors, BPR = generate_best_anchors(args=args, ratio_thres=4, min_size=4)
    print(f'Best Possible Recalls (BPR): {BPR:.4f}')
    print(f'Best Fit Anchors: {str(anchors)}')
    class_list = data_item['NAMES']
    cls_ids = np.concatenate(cls_ids, axis=0)
    fig_cls_dist = visualize_class_dist(cls_ids, class_list, rotation=60)
    fig_cls_dist.savefig(str(args.dist / f'{args.data.name.split(".")[0]}_dist.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='path to data.yaml file')
    parser.add_argument('--config', type=str, default='config/yolov3_coco.yaml', help='path to config.yaml file')
    parser.add_argument('--img_size', type=int, default=416, help='input image size')
    parser.add_argument('--n_cluster', type=int, default=9, help='number of anchors')
    args = parser.parse_args()
    args.data = ROOT / args.data
    args.dist = args.data.parent / 'distribution'
    os.makedirs(args.dist, exist_ok=True)
    main_work(args)


if __name__ == '__main__':
    main()
