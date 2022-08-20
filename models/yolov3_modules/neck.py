import torch
import torch.nn as nn

from element import ConvLayer



class TopDownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 2 == 0
        self.conv1 = ConvLayer(in_channels, out_channels, 1, stride=1, padding=0)
        self.conv2 = ConvLayer(out_channels, out_channels*2, 3, stride=1, padding=1)
        self.conv3 = ConvLayer(out_channels*2, out_channels, 1, stride=1, padding=0)
        self.conv4 = ConvLayer(out_channels, out_channels*2, 3, stride=1, padding=1)
        self.conv5 = ConvLayer(out_channels*2, out_channels, 1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out



class YOLOv3_FPN(nn.Module):
    def __init__(self, last_dim_channels):
        super().__init__()
        self.topdown_1 = TopDownLayer(1024, 512)
        self.topdown_2 = TopDownLayer(768, 256)
        self.topdown_3 = TopDownLayer(384, 128)
        self.lateral_1 = ConvLayer(512, 256, 1, stride=1, padding=0)
        self.lateral_2 = ConvLayer(256, 128, 1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_S = nn.Sequential(
            ConvLayer(128, 128*2, 3, stride=1, padding=1),
            nn.Conv2d(128*2, last_dim_channels, 1, stride=1, padding=0)
        )
        self.conv_M = nn.Sequential(
            ConvLayer(256, 256*2, 3, stride=1, padding=1),
            nn.Conv2d(256*2, last_dim_channels, 1, stride=1, padding=0)
        )
        self.conv_L = nn.Sequential(
            ConvLayer(512, 512*2, 3, stride=1, padding=1),
            nn.Conv2d(512*2, last_dim_channels, 1, stride=1, padding=0)
        )
    
    def forward(self, x):
        C3, C4, C5 = x
        P5 = self.topdown_1(C5)
        ftr_S = self.conv_L(P5)
        C4 = torch.cat((self.upsample(P5), self.lateral_1(C4)), dim=1)
        P4 = self.topdown_2(C4)
        ftr_M = self.conv_M(P4)
        C3 = torch.cat((self.upsample(P4), self.lateral_2(C3)), dim=1)
        P3 = self.topdown_3(C3)
        ftr_L = self.conv_S(P3)
        return [ftr_S, ftr_M, ftr_L]



if __name__ == "__main__":
    from backbone import Darknet53_backbone
    
    x = torch.randn(1, 3, 416, 416)
    num_classes = 80
    num_attribute =  5 + num_classes
    num_anchor_per_scale = 3
    last_dim_channels = num_attribute * num_anchor_per_scale
    backbone = Darknet53_backbone(pretrained=True)
    fpn_module = YOLOv3_FPN(last_dim_channels)

    with torch.no_grad():
        features = backbone(x)
        features = fpn_module(features)

    for feature in features:
        print(feature.shape)