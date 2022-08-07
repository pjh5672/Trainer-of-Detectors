import torch
import torch.nn as nn



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, lrelu_neg_slope = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)
        return out



class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 2 == 0
        half_in_channels = in_channels // 2
        self.conv1 = ConvLayer(in_channels, half_in_channels, 1, stride=1, padding=0)
        self.conv2 = ConvLayer(half_in_channels, in_channels, 3, stride=1, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out