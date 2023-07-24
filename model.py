import numpy as np
import torch
import torch.nn as nn
import config

config1= [
    [6, 2, 2, 64, False],
    [3, 2, 1, 128, False],
    ("c3", 3, [1, 1, 0, 128, False], False, True),
    [3, 2, 1, 256, False],
    ("c3", 6, [1, 1, 0, 256, False], True, True),
    [3, 2, 1, 512, False],
    ("c3", 9, [1, 1, 0, 512, False], True, True),
    [3, 2, 1, 1024, False],
    ("c3", 3, [1, 1, 0, 1024, False], False, True),
    "sppf",
    [1, 1, 0, 512, True],
    "U",
    ("c3", 3, [1, 1, 0, 512, False], False, False),
    [1, 1, 0, 256, True],
    "U",
    ("c3", 3, [1, 1, 0, 256, False], False, False),
    "P",
    [3, 2, 1, 256, False],
    ("c3", 3, [1, 1, 0, 512, False], False, False),
    "P",
    [3, 2, 1, 512, False],
    ("c3", 3, [1, 1, 0, 1024, False], False, False),
    "P",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SpatialPyramidPoolingWithFixedBins(nn.Module):
    def __init__(self, in_channels, pool_sizes=[[5, 1, 2], [5, 1, 2], [5, 1, 2]]):
        super(SpatialPyramidPoolingWithFixedBins, self).__init__()
        self.firstblock = ConvBNSiLUBlock(in_channels , in_channels//2, kernel_size=1, stride=1, padding=0)

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size[0], pool_size[1], pool_size[2]) for pool_size in pool_sizes])
        self.lastblock = ConvBNSiLUBlock(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features=[]
        out=self.firstblock(x)
        features.append(out)

        for maxpool in self.maxpools:
            out=maxpool(out)
            features.append(out)

        features = torch.cat(features, dim=1)
        features = self.lastblock(features)
        return features

class ConvBNSiLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, concat=False):
        super(ConvBNSiLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.concat = concat

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x

class BottleNeckDown(nn.Module):
    def __init__(self, in_channels,):
        super(BottleNeckDown, self).__init__()
        reduced_channels = int(in_channels / 2)
        self.block0 = ConvBNSiLUBlock(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0)
        self.block1 = ConvBNSiLUBlock(reduced_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.block0(x)
        out = self.block1(out)
        return out

class BottleNeckUp(nn.Module):
    def __init__(self, in_channels):
        super(BottleNeckUp, self).__init__()
        reduced_channels = int(in_channels / 2)
        self.block0 = ConvBNSiLUBlock(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0)
        self.block1 = ConvBNSiLUBlock(reduced_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.block0(x)
        out = self.block1(out)
        out += residual
        return out

class C3(nn.Module):
    def __init__(self, in_channels, out_channels, nblocks, kernel_size, stride, padding, concat=False, BtNUp=True):
        super(C3, self).__init__()
        reduced_channels = int(in_channels / 2)
        self.block0= ConvBNSiLUBlock(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0)
        self.block1 = ConvBNSiLUBlock(in_channels, reduced_channels, kernel_size=3, stride=1, padding=1)
        cspblocks = []
        self.concat = concat
        for i in range(nblocks):
            if BtNUp:
                cspblocks.append(BottleNeckUp(in_channels//2))
            else:
                cspblocks.append(BottleNeckDown(in_channels//2))

        self.cspblocks = nn.Sequential(*cspblocks)
        self.lastblock = ConvBNSiLUBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        split0 = self.block0(x)
        split1 = self.block1(x)
        blocks = self.cspblocks(split0)
        out = torch.cat([split1, blocks], dim=1)
        out=self.lastblock(out)
        return out

class predict(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(predict, self).__init__()
        self.conv = nn.Conv2d(in_channels, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0)
        self.num_classes = num_classes

    def forward(self, resb):
        out = self.conv(resb)
        return out.reshape(out.shape[0], 3, self.num_classes + 5, out.shape[2], out.shape[3]).permute(0, 1, 3, 4, 2)


class Yolov5(nn.Module):
    def __init__(self, config: list ,n_channels=3, num_classes=config.NUM_CLASSES):
        super(Yolov5, self).__init__()
        self.config=config
        self.n_channels=n_channels
        self.num_classes=num_classes
        self.layers = self._create_model()

    def forward(self, x):
        outputs = []  # for each scale
        route_connectionsup = []
        route_connectionsdown = []
        upconcat = False
        for layer in self.layers:
            if isinstance(layer, predict):
                outputs.append(layer(x))
                upconcat = True
                continue

            x = layer(x)

            if isinstance(layer, C3) and layer.concat:
                route_connectionsdown.append(x)
            elif isinstance(layer, ConvBNSiLUBlock) and layer.concat:
                route_connectionsup.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connectionsdown[-1]], dim=1)
                route_connectionsdown.pop()
            elif isinstance(layer, ConvBNSiLUBlock) and upconcat:
                x = torch.cat([x, route_connectionsup[-1]], dim=1)
                upconcat = False
                route_connectionsup.pop()
        return outputs[::-1]
    def _create_model(self):
        blocks=[]
        in_channels=self.n_channels
        for idx, module in enumerate(self.config):
            if isinstance(module,list):
                kernel_size, stride, padding, out_channels, concat = module
                blocks.append(ConvBNSiLUBlock(in_channels, out_channels, kernel_size, stride, padding, concat=concat))
                if self.config[idx-1] == "P" and idx>1:
                    in_channels = out_channels * 2
                else : in_channels = out_channels
            if isinstance(module, tuple):
                type, nblocks, (kernel_size, stride, padding, out_channels, concatconv), concat, BtNUp = module
                blocks.append(C3(in_channels, out_channels, nblocks, kernel_size, stride, padding, concat=concat, BtNUp=BtNUp))
                in_channels = out_channels
            if isinstance(module, str):
                if module == "sppf":
                    blocks.append(SpatialPyramidPoolingWithFixedBins(in_channels))
                elif module == "U":
                    blocks.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 2
                elif module == "P":
                    blocks.append(predict(in_channels=in_channels, num_classes=self.num_classes))
        return nn.Sequential(*blocks)


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 640
    # Usage example:
    import time
    Yv5 =Yolov5(config=config,num_classes=num_classes).to(DEVICE)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
    s=time.perf_counter()
    out = Yv5(x)
    assert Yv5(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert Yv5(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert Yv5(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print(f"Success!, Total time: {time.perf_counter()-s}",)
