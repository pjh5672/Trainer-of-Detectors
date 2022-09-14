import yaml
import torch
import torch.nn as nn

from yolov3_modules import Darknet53_backbone, YOLOv3_FPN, YOLOv3_head



class YOLOv3_Model(nn.Module):
    def __init__(self, config_path, num_classes):
        super().__init__()
        with open(config_path) as f:
            item = yaml.load(f, Loader=yaml.FullLoader)

        self.input_size = item['INPUT_SIZE']
        self.anchors = [x for x in item['ANCHORS'].values()]
        self.num_classes = num_classes
        self.num_attribute = 5 + self.num_classes
        self.num_anchor_per_scale = len(self.anchors[0])
        self.last_dim_channels = self.num_attribute * self.num_anchor_per_scale
        self.backbone = Darknet53_backbone()
        self.fpn = YOLOv3_FPN()
        self.head = YOLOv3_head(input_size=self.input_size, num_classes=num_classes, anchors=self.anchors)
        self.apply(self._weight_init_kaiming_uniform)


    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        out_l, out_m, out_s = self.fpn(x1, x2, x3)
        predictions = self.head(out_l, out_m, out_s)
        return predictions


    def _weight_init_kaiming_uniform(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)



if __name__ == "__main__":
    from pathlib import Path

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]

    config_path = ROOT / 'config' / 'yolov3_coco.yaml'
    num_classes = 80
    device = torch.device('cpu')

    model = YOLOv3_Model(config_path=config_path, num_classes=num_classes)
    model = model.to(device)
    model.eval()
    x = torch.randn(2, 3, 416, 416).to(device)
    with torch.no_grad():
        predictions = model(x)

    for prediction in predictions:
        print(prediction.shape)