import yaml
import torch
import torch.nn as nn
import numpy as np


class YOLOv3_Loss():
    def __init__(self, config_path, model):
        super().__init__()

        with open(config_path) as f:
            item = yaml.load(f, Loader=yaml.FullLoader)

        self.input_size = item['INPUT_SIZE']
        self.ignore_threshold = item['IGNORE_THRESH']
        self.coeff_coord = item['COEFFICIENT_COORD']
        self.coeff_noobj = item['COEFFICIENT_NOOBJ']

        self.device = model.device
        self.num_classes = model.head.num_classes
        self.num_anchor_per_scale = model.head.num_anchor_per_scale

        dummy_x = torch.randn(1, 3, self.input_size, self.input_size).to(self.device) 
        with torch.no_grad():
            _ = model(dummy_x)

        self.anchors = [model.head.anchor_L, model.head.anchor_M, model.head.anchor_S]
        self.strides = [model.head.head_L.stride, model.head.head_M.stride, model.head.head_S.stride]
        self.num_anchors = len(self.anchors)
        self.mae_loss = nn.L1Loss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')


    def __call__(self, predictions, targets):
        coord_loss, obj_loss, noobj_loss, cls_loss, total_loss = 0., 0., 0., 0., 0.

        for scale_index in range(self.num_anchors):
            anchor_each_scale = self.anchors[scale_index]
            stride_each_scale = self.strides[scale_index]
            prediction_each_scale = predictions[scale_index]
            self.batch_size, num_preds, _ = prediction_each_scale.shape
            self.grid_size = int(np.sqrt(num_preds/self.num_anchor_per_scale))

            prediction_each_scale = prediction_each_scale.view(self.batch_size, self.num_anchor_per_scale, self.grid_size, self.grid_size, -1)
            pred_box = prediction_each_scale[..., :4]
            pred_obj = prediction_each_scale[..., 4]
            pred_cls = prediction_each_scale[..., 5:]

            pred_tx, pred_ty, pred_tw, pred_th = self.transfrom_batch_pred_loss_form(pred_box, anchor_each_scale, stride_each_scale)
            b_obj_mask, b_noobj_mask, b_target_tx, b_target_ty, b_target_tw, b_target_th, b_target_cls = self.transform_batch_target_loss_form(targets, anchor_each_scale)
            b_target_obj = b_obj_mask.float()
            b_obj_mask = b_obj_mask.type(torch.BoolTensor)
            b_noobj_mask = b_noobj_mask.type(torch.BoolTensor)

            loss_tx = self.mae_loss(pred_tx[b_obj_mask], b_target_tx[b_obj_mask])
            loss_ty = self.mae_loss(pred_ty[b_obj_mask], b_target_ty[b_obj_mask])
            loss_tw = self.mae_loss(pred_tw[b_obj_mask], b_target_tw[b_obj_mask])
            loss_th = self.mae_loss(pred_th[b_obj_mask], b_target_th[b_obj_mask])
            loss_obj = self.bce_loss(pred_obj[b_obj_mask], b_target_obj[b_obj_mask])
            loss_noobj = self.bce_loss(pred_obj[b_noobj_mask], b_target_obj[b_noobj_mask])
            loss_cls = self.bce_loss(pred_cls[b_obj_mask], b_target_cls[b_obj_mask])
            
            coord_loss += (loss_tx + loss_ty + loss_tw + loss_th)
            obj_loss += loss_obj
            noobj_loss += loss_noobj
            cls_loss += loss_cls
            total_loss += self.coeff_coord * (loss_tx + loss_ty + loss_tw + loss_th) + \
                            loss_obj + self.coeff_noobj * loss_noobj + loss_cls
            
        coord_loss /= self.batch_size
        obj_loss /= self.batch_size
        noobj_loss /= self.batch_size
        cls_loss /= self.batch_size
        total_loss /= self.batch_size
        return total_loss, coord_loss, obj_loss, noobj_loss, cls_loss


    def transfrom_batch_pred_loss_form(self, prediction_box, anchor_each_scale, stride_each_scale):
        anchor_w = anchor_each_scale[:, 0].view((1, self.num_anchor_per_scale, 1, 1))
        anchor_h = anchor_each_scale[:, 1].view((1, self.num_anchor_per_scale, 1, 1))

        bx = prediction_box[..., 0] / stride_each_scale
        by = prediction_box[..., 1] / stride_each_scale
        bw = prediction_box[..., 2]
        bh = prediction_box[..., 3] 

        tx = bx - bx.floor()
        ty = by - by.floor()
        tw = torch.log(bw / anchor_w)
        th = torch.log(bh / anchor_h)
        return tx, ty, tw, th


    def get_IoU_target_with_anchor(self, wh1, wh2):
        w1, h1 = wh1
        w2, h2 = wh2
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / union_area


    def build_target_mask(self, grid_ij, target_xy, target_wh, anchor_each_scale):
        obj_mask = torch.zeros(size=(self.num_anchor_per_scale, self.grid_size, self.grid_size), 
                               device=self.device, dtype=torch.uint8)
        noobj_mask = torch.ones(size=(self.num_anchor_per_scale, self.grid_size, self.grid_size), 
                                device=self.device, dtype=torch.uint8)
        iou_target_with_anchor = [self.get_IoU_target_with_anchor(target_wh.t(), anchor) for anchor in anchor_each_scale]
        iou_target_with_anchor = torch.stack(iou_target_with_anchor, dim=0)
        best_iou, best_anchor_index = iou_target_with_anchor.max(dim=0)

        obj_mask[best_anchor_index, grid_ij[1], grid_ij[0]] = 1
        noobj_mask[best_anchor_index, grid_ij[1], grid_ij[0]] = 0

        for index, iou in enumerate(iou_target_with_anchor.t()):
            noobj_mask[iou > self.ignore_threshold, grid_ij[1][index], grid_ij[0][index]] = 0

        return best_anchor_index, obj_mask, noobj_mask


    def transform_target_loss_form(self, target, anchor_each_scale):
        target_tx = torch.zeros(size=(self.num_anchor_per_scale, self.grid_size, self.grid_size), 
                                device=self.device, dtype=torch.float32)
        target_ty = torch.zeros(size=(self.num_anchor_per_scale, self.grid_size, self.grid_size), 
                                device=self.device, dtype=torch.float32)
        target_tw = torch.zeros(size=(self.num_anchor_per_scale, self.grid_size, self.grid_size), 
                                device=self.device, dtype=torch.float32)
        target_th = torch.zeros(size=(self.num_anchor_per_scale, self.grid_size, self.grid_size), 
                                device=self.device, dtype=torch.float32)
        target_cls = torch.zeros(size=(self.num_anchor_per_scale, self.grid_size, self.grid_size, self.num_classes), 
                                device=self.device, dtype=torch.float32)

        target_c = target[:, 0].long()
        if (target_c == -1).any():
            obj_mask = torch.zeros(size=(self.num_anchor_per_scale, self.grid_size, self.grid_size), 
                                   device=self.device, dtype=torch.uint8)
            noobj_mask = torch.ones(size=(self.num_anchor_per_scale, self.grid_size, self.grid_size), 
                                    device=self.device, dtype=torch.uint8)
            return obj_mask, noobj_mask, target_tx, target_ty, target_tw, target_th, target_cls

        target_xy = target[:, 1:3] * self.grid_size
        target_wh = target[:, 3:5] * self.input_size
        grid_ij = target_xy.long().t()

        best_anchor_index, obj_mask, noobj_mask = self.build_target_mask(grid_ij, target_xy, target_wh, anchor_each_scale)
        anchor_wh = anchor_each_scale[best_anchor_index]

        target_tx[best_anchor_index, grid_ij[1], grid_ij[0]] = target_xy[:, 0] - target_xy[:, 0].floor()
        target_ty[best_anchor_index, grid_ij[1], grid_ij[0]] = target_xy[:, 1] - target_xy[:, 1].floor()
        target_tw[best_anchor_index, grid_ij[1], grid_ij[0]] = torch.log(target_wh[:, 0] / anchor_wh[:, 0])
        target_th[best_anchor_index, grid_ij[1], grid_ij[0]] = torch.log(target_wh[:, 1] / anchor_wh[:, 1])
        target_cls[best_anchor_index, grid_ij[1], grid_ij[0], target_c] = 1
        return obj_mask, noobj_mask, target_tx, target_ty, target_tw, target_th, target_cls


    def transform_batch_target_loss_form(self, targets, anchor_each_scale):
        b_obj_mask, b_noobj_mask, b_target_tx, b_target_ty, b_target_tw, b_target_th, b_target_cls= [], [], [], [], [], [], []

        for target in targets:
            target = torch.from_numpy(target).to(self.device)
            target_loss_form = self.transform_target_loss_form(target, anchor_each_scale)
            obj_mask, noobj_mask, target_tx, target_ty, target_tw, target_th, target_cls = target_loss_form

            b_obj_mask.append(obj_mask)
            b_noobj_mask.append(noobj_mask)
            b_target_tx.append(target_tx)
            b_target_ty.append(target_ty)
            b_target_tw.append(target_tw)
            b_target_th.append(target_th)
            b_target_cls.append(target_cls)

        b_obj_mask = torch.stack(b_obj_mask, dim=0)
        b_noobj_mask = torch.stack(b_noobj_mask, dim=0)
        b_target_tx = torch.stack(b_target_tx, dim=0)
        b_target_ty = torch.stack(b_target_ty, dim=0)
        b_target_tw = torch.stack(b_target_tw, dim=0)
        b_target_th = torch.stack(b_target_th, dim=0)
        b_target_cls = torch.stack(b_target_cls, dim=0)
        return b_obj_mask, b_noobj_mask, b_target_tx, b_target_ty, b_target_tw, b_target_th, b_target_cls




if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from dataloader import build_dataloader
    from models import YOLOv3_Model

    data_path = ROOT / 'data' / 'coco128.yml'
    config_path = ROOT / 'config' / 'yolov3.yml'

    with open(config_path) as f:
        item = yaml.load(f, Loader=yaml.FullLoader)

    input_size = item['INPUT_SIZE']
    batch_size = 2
    device = torch.device('cpu')

    dataloaders, classname_list = build_dataloader(data_path=data_path, image_size=(input_size, input_size), batch_size=batch_size)
    model = YOLOv3_Model(config_path=config_path, num_classes=len(classname_list), device=device, pretrained=True)
    model = model.to(device)

    criterion = YOLOv3_Loss(config_path=config_path, model=model)

    # sanity check
    phase = 'train'
    for index, minibatch in enumerate(dataloaders[phase]):
        images = minibatch[0].to(device)
        targets = minibatch[1]
        filenames = minibatch[2]
        
        if index % 2 == 0:
            predictions = model(images)
            total_loss, coord_loss, obj_loss, noobj_loss, cls_loss = criterion(predictions, targets)
            print(total_loss, coord_loss, obj_loss, noobj_loss, cls_loss)
            total_loss.backward()
            break