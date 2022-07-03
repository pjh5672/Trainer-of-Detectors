from pathlib import Path

import torch
import torch.nn as nn

from element import ConvLayer, ResBlock

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


class Darknet53_backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, 3, stride=1, padding=1)
        self.res_block1 = self._build_Conv_and_ResBlock(32, 64, 1)
        self.res_block2 = self._build_Conv_and_ResBlock(64, 128, 2)
        self.res_block3 = self._build_Conv_and_ResBlock(128, 256, 8)
        self.res_block4 = self._build_Conv_and_ResBlock(256, 512, 8)
        self.res_block5 = self._build_Conv_and_ResBlock(512, 1024, 4)

        if pretrained:
            ckpt = torch.load(ROOT / 'darknet53_features.pth')
            self.load_state_dict(ckpt, strict=True)
        else:
            self.apply(self._weight_init_xavier_uniform)


    def forward(self, x):
        out = self.conv1(x)
        C1 = self.res_block1(out)
        C2 = self.res_block2(C1)
        C3 = self.res_block3(C2)
        C4 = self.res_block4(C3)
        C5 = self.res_block5(C4)
        return [C3, C4, C5]


    def _build_Conv_and_ResBlock(self, in_channels, out_channels, num_block):
        model = nn.Sequential()
        model.add_module("conv", ConvLayer(in_channels, out_channels, 3, stride=2, padding=1))
        for idx in range(num_block):
            model.add_module(f"res{idx}", ResBlock(out_channels))
        return model


    def _weight_init_xavier_uniform(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.fill_(1.0)



if __name__ == "__main__":
    x = torch.randn(1, 3, 416, 416)
    backbone = Darknet53_backbone(pretrained=True)
    backbone.eval()
    with torch.no_grad():
        features = backbone(x)
    
    for feature in features:
        print(feature.shape)