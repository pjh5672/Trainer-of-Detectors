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
        self.batch_size = item['BATCH_SIZE']
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

        self.mae_loss = nn.L1Loss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')


    def __call__(self, predictions, targets):
        coord_loss, obj_loss, cls_loss, noobj_loss = 0., 0., 0., 0.

        for scale in range(self.num_anchor_per_scale):
            prediction_each_scale = predictions[scale]
            anchor_each_scale = self.anchors[scale]
            stride_each_scale = self.strides[scale]

            self.batch_size, num_preds, _ = prediction_each_scale.shape
            self.grid_size = int(np.sqrt(num_preds/self.num_anchor_per_scale))
            
            batch_pred_loss_form = self.transfrom_batch_pred_loss_form(prediction_each_scale, anchor_each_scale, stride_each_scale)
            batch_target_loss_form, batch_target_noobj_loss_form = self.transform_batch_target_loss_form(targets, anchor_each_scale)

            object_mask = batch_target_loss_form[..., 4].eq(1.)
            noobject_mask = batch_target_noobj_loss_form.eq(1.)

            coord_loss += self.mae_loss(batch_pred_loss_form[..., :4][object_mask], batch_target_loss_form[..., :4][object_mask])
            obj_loss += self.bce_loss(batch_pred_loss_form[..., 4][object_mask], batch_target_loss_form[..., 4][object_mask])
            cls_loss += self.bce_loss(batch_pred_loss_form[..., 5:][object_mask], batch_target_loss_form[..., 5:][object_mask])
            noobj_loss += self.bce_loss(batch_pred_loss_form[..., 4][noobject_mask], batch_target_noobj_loss_form[noobject_mask])

        coord_loss /= self.batch_size
        obj_loss /= self.batch_size
        cls_loss /= self.batch_size
        noobj_loss /= self.batch_size
        total_loss = self.coeff_coord * coord_loss + obj_loss + cls_loss + self.coeff_noobj * noobj_loss
        return total_loss, coord_loss, obj_loss, cls_loss, noobj_loss


    def transfrom_batch_pred_loss_form(self, prediction_each_scale, anchor_each_scale, stride_each_scale):
        prediction_each_scale = prediction_each_scale.view(self.batch_size, self.num_anchor_per_scale, self.grid_size, self.grid_size, -1)
        predictions_form = torch.zeros_like(prediction_each_scale)
        anchor_w = anchor_each_scale[:, 0].view((1, self.num_anchor_per_scale, 1, 1))
        anchor_h = anchor_each_scale[:, 1].view((1, self.num_anchor_per_scale, 1, 1))

        bx = prediction_each_scale[..., 0] / stride_each_scale
        by = prediction_each_scale[..., 1] / stride_each_scale
        bw = prediction_each_scale[..., 2]
        bh = prediction_each_scale[..., 3] 

        tx = bx - bx.floor()
        ty = by - by.floor()
        tw = torch.log(bw / anchor_w)
        th = torch.log(bh / anchor_h)
        
        predictions_form[..., :4] = torch.stack([tx, ty, tw, th], dim=-1)
        predictions_form[..., 4] = prediction_each_scale[..., 4]
        predictions_form[..., 5:] = prediction_each_scale[..., 5:] * prediction_each_scale[..., [4]]
        return predictions_form


    def get_IoU_target_with_anchor(self, wh1, wh2):
        w1, h1 = wh1
        w2, h2 = wh2
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / union_area


    def transform_target_loss_form(self, target, anchor_each_scale):
        target_form = torch.zeros((self.num_anchor_per_scale, self.grid_size, self.grid_size, 5+self.num_classes), device=self.device)
        target_noobj_form = torch.ones((self.num_anchor_per_scale, self.grid_size, self.grid_size), device=self.device)

        for gt in target:
            cls_id, xc, yc, w, h = gt

            if cls_id == -1:
                continue

            grid_x = xc * self.grid_size
            grid_y = yc * self.grid_size
            tx = grid_x - int(grid_x)
            ty = grid_y - int(grid_y)
            
            iou_with_anchor = [self.get_IoU_target_with_anchor([w*self.input_size, h*self.input_size], anchor) for anchor in anchor_each_scale]
            iou_with_anchor = torch.stack(iou_with_anchor, dim=0)
            _, best_anchor_index = iou_with_anchor.max(dim=0)
            
            target_form[best_anchor_index, int(grid_y), int(grid_x), :4] = torch.Tensor([tx,ty,w,h])
            target_form[best_anchor_index, int(grid_y), int(grid_x), 4] = 1.
            target_form[best_anchor_index, int(grid_y), int(grid_x), 5 + cls_id.long()] = 1.

            target_noobj_form[best_anchor_index, int(grid_y), int(grid_x)] = 0.
            target_noobj_form[iou_with_anchor >= self.ignore_threshold, int(grid_y), int(grid_x)] = 0.
        return target_form, target_noobj_form


    def transform_batch_target_loss_form(self, targets, anchor_each_scale):
        batch_target_loss_form = []
        batch_target_noobj_loss_form = []

        for target in targets:
            target = torch.from_numpy(target)
            target_form, target_noobj_form = self.transform_target_loss_form(target, anchor_each_scale)
            batch_target_loss_form.append(target_form)
            batch_target_noobj_loss_form.append(target_noobj_form)

        batch_target_loss_form = torch.stack(batch_target_loss_form, dim=0)
        batch_target_noobj_loss_form = torch.stack(batch_target_noobj_loss_form, dim=0)
        return batch_target_loss_form, batch_target_noobj_loss_form




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
    batch_size = item['BATCH_SIZE']
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
            print(images.shape)
            print(targets)
            predictions = model(images)
            total_loss, coord_loss, obj_loss, cls_loss, noobj_loss = criterion(predictions, targets)
            print(total_loss)
            total_loss.backward()
            break

