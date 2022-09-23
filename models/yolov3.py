import yaml
import torch
import torch.nn as nn

from yolov3_modules import Darknet53, YOLOv3_FPN, YOLOv3_head



class YOLOv3_Model(nn.Module):
    def __init__(self, input_size, anchors, num_classes, freeze_backbone=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_attribute = 5 + num_classes
        self.num_anchor_per_scale = len(anchors[0])
        self.last_dim_channels = self.num_attribute * self.num_anchor_per_scale
        self.backbone = Darknet53(freeze_grad=freeze_backbone)
        self.fpn = YOLOv3_FPN()
        self.head = YOLOv3_head(input_size=input_size, num_classes=num_classes, anchors=anchors)
        

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        out_l, out_m, out_s = self.fpn(x1, x2, x3)
        predictions = self.head(out_l, out_m, out_s)
        return predictions



if __name__ == "__main__":
    import yaml
    from pathlib import Path

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]

    with open(ROOT / 'config' / 'yolov3_coco.yaml') as f:
        item = yaml.load(f, Loader=yaml.FullLoader)

    input_size = item['INPUT_SIZE']
    num_classes = 80
    anchors = list(item['ANCHORS'].values())
    device = torch.device('cpu')

    model = YOLOv3_Model(input_size=input_size, anchors=anchors, num_classes=num_classes)
    model = model.to(device)
    model.eval()
    x = torch.randn(2, 3, input_size, input_size).to(device)
    with torch.no_grad():
        predictions = model(x)

    for prediction in predictions:
        print(prediction.shape)