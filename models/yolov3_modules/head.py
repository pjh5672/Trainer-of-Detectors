import torch
import torch.nn as nn




class DetectLayer(nn.Module):
    def __init__(self, input_size, num_classes, anchors, num_anchor_per_scale):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_attribute =  5 + num_classes
        self.num_anchor_per_scale = num_anchor_per_scale
        self.anchor_w = anchors[:, 0].view((1, self.num_anchor_per_scale, 1, 1))
        self.anchor_h = anchors[:, 1].view((1, self.num_anchor_per_scale, 1, 1))


    def forward(self, x):
        self.device = x.device
        batch_size = x.shape[0]
        grid_size = x.shape[2]
        self.stride = self.input_size / grid_size

        out = x.view(batch_size, self.num_anchor_per_scale, self.num_attribute, grid_size, grid_size)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        grid_xy = self.compute_grid_offset(grid_size)
        pred_bboxes = self.transform_pred_coords(bboxes=out[..., :4], grid_xy=grid_xy)
        pred_object = out[..., 4]
        pred_class = out[..., 5:]
        prediction = torch.cat((pred_bboxes.view(batch_size, -1, 4),
                                pred_object.view(batch_size, -1, 1),
                                pred_class.view(batch_size, -1, self.num_classes)),
                                dim=-1)
        return prediction

    
    def compute_grid_offset(self, grid_size):
        grid_offset = torch.arange(grid_size).to(self.device)
        grid_y, grid_x = torch.meshgrid(grid_offset, grid_offset, indexing='ij')
        grid_xy = [grid_x, grid_y]
        return grid_xy

    
    def transform_pred_coords(self, bboxes, grid_xy):
        bx = (torch.sigmoid(bboxes[..., 0]) + grid_xy[0]) * self.stride
        by = (torch.sigmoid(bboxes[..., 1]) + grid_xy[1]) * self.stride
        bw = torch.exp(bboxes[..., 2]) * self.anchor_w.to(self.device)
        bh = torch.exp(bboxes[..., 3]) * self.anchor_h.to(self.device)
        pred_bboxes = torch.stack([bx, by, bw, bh], dim=-1)
        return pred_bboxes



class YOLOv3_head(nn.Module):
    def __init__(self, input_size, num_classes, anchors, num_anchor_per_scale):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.anchor_scale = self.input_size / 416
        self.num_anchor_per_scale = num_anchor_per_scale
        self.anchor_L = torch.Tensor(anchors[2]) * self.anchor_scale
        self.anchor_M = torch.Tensor(anchors[1]) * self.anchor_scale
        self.anchor_S = torch.Tensor(anchors[0]) * self.anchor_scale

        self.head_L = DetectLayer(input_size=self.input_size,
                                  num_classes=self.num_classes,
                                  anchors=self.anchor_L,
                                  num_anchor_per_scale=self.num_anchor_per_scale)

        self.head_M = DetectLayer(input_size=self.input_size,
                                  num_classes=self.num_classes,
                                  anchors=self.anchor_M,
                                  num_anchor_per_scale=self.num_anchor_per_scale)

        self.head_S = DetectLayer(input_size=self.input_size,
                                  num_classes=self.num_classes,
                                  anchors=self.anchor_S,
                                  num_anchor_per_scale=self.num_anchor_per_scale)

    def forward(self, x1, x2, x3):
        pred_L = self.head_L(x1)
        pred_M = self.head_M(x2)
        pred_S = self.head_S(x3)
        return [pred_L, pred_M, pred_S]



if __name__ == "__main__":
    import yaml
    from pathlib import Path

    from backbone import Darknet53_backbone
    from neck import YOLOv3_FPN

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]

    with open(ROOT / 'config' / 'yolov3.yml') as f:
        item = yaml.load(f, Loader=yaml.FullLoader)

    input_size = item['INPUT_SIZE']
    anchors = [x for x in item['ANCHORS'].values()]
    num_classes = 80
    device = torch.device('cpu')

    x = torch.randn(1, 3, 416, 416).to(device)
    num_classes = 80
    num_attribute =  5 + num_classes
    num_anchor_per_scale = 3
    last_dim_channels = num_attribute * num_anchor_per_scale
    backbone = Darknet53_backbone().to(device)
    fpn = YOLOv3_FPN(last_dim_channels=last_dim_channels).to(device)
    head = YOLOv3_head(input_size=input_size, num_classes=80, anchors=anchors, num_anchor_per_scale=num_anchor_per_scale)

    with torch.no_grad():
        x1, x2, x3 = backbone(x)
        out_l, out_m, out_s = fpn(x1, x2, x3)
        predictions = head(out_l, out_m, out_s)

    for prediction in predictions:
        print(prediction.shape)