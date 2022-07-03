import yaml
import torch.nn as nn

from yolov3_modules import Darknet53_backbone, YOLOv3_FPN, YOLOv3_head



class YOLOv3_Model(nn.Module):
    def __init__(self, config_path, num_classes, device, pretrained=True):
        super().__init__()
    
        with open(config_path) as f:
            item = yaml.load(f, Loader=yaml.FullLoader)

        input_size = item['INPUT_SIZE']
        anchors = [x for x in item['ANCHORS'].values()]
        self.device = device
        self.backbone = Darknet53_backbone(pretrained=pretrained)
        self.fpn = YOLOv3_FPN()
        self.head = YOLOv3_head(input_size=input_size, 
                                num_classes=num_classes, 
                                anchors=anchors, 
                                device=self.device)


    def forward(self, x):
        features = self.backbone(x)
        features = self.fpn(features)
        predictions = self.head(features)
        return predictions


if __name__ == "__main__":
    from pathlib import Path
    import torch

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]

    config_path = ROOT / 'config' / 'yolov3.yml'
    num_classes = 80
    device = torch.device('cpu')

    model = YOLOv3_Model(config_path=config_path,
                        num_classes=num_classes,
                        device=device,
                        pretrained=True)
    model = model.to(device)

    x = torch.randn(1, 3, 416, 416).to(device)
    with torch.no_grad():
        predictions = model(x)

    for prediction in predictions:
        print(prediction.shape)