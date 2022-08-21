import torch
import torch.nn as nn

from element import ConvLayer


class TopDownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, last_dim_channels):
        super().__init__()
        assert out_channels % 2 == 0  #assert out_channels is an even number
        half_out_channels = out_channels // 2
        self.conv1 = ConvLayer(in_channels, half_out_channels, 1, stride=1, padding=0)
        self.conv2 = ConvLayer(half_out_channels, out_channels, 3, stride=1, padding=1)
        self.conv3 = ConvLayer(out_channels, half_out_channels, 1, stride=1, padding=0)
        self.conv4 = ConvLayer(half_out_channels, out_channels, 3, stride=1, padding=1)
        self.conv5 = ConvLayer(out_channels, half_out_channels, 1, stride=1, padding=0)
        self.conv6 = ConvLayer(half_out_channels, out_channels, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(out_channels, last_dim_channels, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        self.branch = self.conv5(tmp)
        tmp = self.conv6(self.branch)
        out = self.conv7(tmp)
        return out


class YOLOv3_FPN(nn.Module):
    def __init__(self, last_dim_channels):
        super().__init__()
        self.topdown1 = TopDownLayer(1024, 1024, last_dim_channels)
        self.conv1 = ConvLayer(512, 256, 1, stride=1, padding=0)
        self.topdown2 = TopDownLayer(768, 512, last_dim_channels)
        self.conv2 = ConvLayer(256, 128, 1, stride=1, padding=0)
        self.topdown3 = TopDownLayer(384, 256, last_dim_channels)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2, x3):
        out1 = self.topdown1(x1)
        tmp = self.upsample(self.conv1(self.topdown1.branch))
        tmp = torch.cat((tmp, x2), dim=1)
        out2 = self.topdown2(tmp)
        tmp = self.upsample(self.conv2(self.topdown2.branch))
        tmp = torch.cat((tmp, x3), dim=1)
        out3 = self.topdown3(tmp)
        return out1, out2, out3


if __name__ == "__main__":
    from backbone import Darknet53_backbone
    
    x = torch.randn(1, 3, 416, 416)
    num_classes = 80
    num_attribute =  5 + num_classes
    num_anchor_per_scale = 3
    last_dim_channels = num_attribute * num_anchor_per_scale
    backbone = Darknet53_backbone()
    fpn_module = YOLOv3_FPN(last_dim_channels)

    with torch.no_grad():
        x1, x2, x3 = backbone(x)
        features = fpn_module(x1, x2, x3)

    for feature in features:
        print(feature.shape)