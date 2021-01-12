import torch
import torch.nn as nn

class Conv_Batch_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv_Batch_Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

def conv_batch_layer(in_channels, out_channels, kernel_size, padding=1, stride=1):
    conv_batch_layer = Conv_Batch_Layer(in_channels, out_channels, kernel_size, stride, padding)
    return conv_batch_layer

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1x1 = conv_batch_layer(in_channels, in_channels//2, 1)
        self.conv3x3 = conv_batch_layer(in_channels//2, in_channels, 3, padding=0)
    
    def forward(self, x):
        out = self.conv1x1(x)
        out = self.conv3x3(out)
        return x + out

def resblock(in_channels):
    residual_block = ResBlock(in_channels)
    return residual_block

class Darknet53(nn.Module):
    def __init__(self, block_nums=[1, 2, 8, 8, 4], block_channels=[64, 128, 256, 512, 1024], in_channels=3):
        super(Darknet53, self).__init__()
        assert (len(block_nums) == len(block_channels))
        self.block_nums = block_nums
        self.in_channels_list = block_channels
        self.top_layer = conv_batch_layer(in_channels, 32, 3, 1)
        self.res_unit = self.resoperator()

    def resoperator(self):
        layers = []
        for block, in_channels in zip(self.block_nums, self.in_channels_list):
            layers.append(conv_batch_layer(in_channels // 2, in_channels, 3, 1, 2))
            for _ in range(block):
                layers.append(resblock(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.top_layer(x)
        x = self.res_unit(x)
        return x

def darknet53():
    darknet = Darknet53()
    return darknet

if __name__ == "__main__":
    darknet = darknet53()
    x = torch.randn(10, 3, 416, 416)
    output = darknet(x)
