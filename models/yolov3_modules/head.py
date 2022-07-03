import torch
import torch.nn as nn

from element import ConvLayer



class DetectLayer(nn.Module):
    def __init__(self, input_size, in_channels, num_classes, anchors, num_anchor_per_scale, device):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_anchor_per_scale = num_anchor_per_scale
        self.num_attribute =  5 + num_classes
        self.last_dim_channels = self.num_attribute * self.num_anchor_per_scale
        self.anchor_w = anchors[:, 0].view((1, self.num_anchor_per_scale, 1, 1))
        self.anchor_h = anchors[:, 1].view((1, self.num_anchor_per_scale, 1, 1))

        self.conv = nn.Sequential(
            ConvLayer(in_channels, in_channels*2, 3, stride=1, padding=1),
            nn.Conv2d(in_channels*2, self.last_dim_channels, 1, stride=1, padding=0)
        ).to(self.device)


    def forward(self, x):
        batch_size = x.shape[0]
        grid_size = x.shape[2]
        self.stride = self.input_size / grid_size

        out = self.conv(x)
        out = out.view(batch_size, self.num_anchor_per_scale, self.num_attribute, grid_size, grid_size)
        out = out.permute(0, 1, 3, 4, 2).contiguous()

        grid_xy = self.compute_grid_offset(grid_size)
        pred_bboxes = self.transform_pred_coords(bboxes=out[..., :4], grid_xy=grid_xy)
        pred_object = torch.sigmoid(out[..., 4])
        pred_class = torch.sigmoid(out[..., 5:])

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
        bw = torch.exp(bboxes[..., 2]) * self.anchor_w
        bh = torch.exp(bboxes[..., 3]) * self.anchor_h
        pred_bboxes = torch.stack([bx, by, bw, bh], dim=-1)
        return pred_bboxes



class YOLOv3_head(nn.Module):
    def __init__(self, input_size, num_classes, anchors, device):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_anchor_per_scale = len(anchors[2])
        self.anchor_L = torch.Tensor(anchors[2]).to(device)
        self.anchor_M = torch.Tensor(anchors[1]).to(device)
        self.anchor_S = torch.Tensor(anchors[0]).to(device)
        
        self.head_L = DetectLayer(input_size=self.input_size,
                                  in_channels=512,
                                  num_classes=self.num_classes,
                                  anchors=self.anchor_L,
                                  num_anchor_per_scale=self.num_anchor_per_scale,
                                  device=device)

        self.head_M = DetectLayer(input_size=self.input_size,
                                  in_channels=256,
                                  num_classes=self.num_classes,
                                  anchors=self.anchor_M,
                                  num_anchor_per_scale=self.num_anchor_per_scale,
                                  device=device)

        self.head_S = DetectLayer(input_size=self.input_size,
                                  in_channels=128,
                                  num_classes=self.num_classes,
                                  anchors=self.anchor_S,
                                  num_anchor_per_scale=self.num_anchor_per_scale,
                                  device=device)
        

    def forward(self, x):
        P3, P4, P5 = x
        pred_L = self.head_L(P5)
        pred_M = self.head_M(P4)
        pred_S = self.head_S(P3)
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
    backbone = Darknet53_backbone(pretrained=True).to(device)
    fpn = YOLOv3_FPN().to(device)
    head = YOLOv3_head(input_size=input_size, num_classes=80, anchors=anchors, device=device)

    with torch.no_grad():
        features = backbone(x)
        features = fpn(features)
        predictions = head(features)

    for prediction in predictions:
        print(prediction.shape)
    