import torch
import torch.nn as nn

config=[
    [6, 2, 2, 64],
    [3, 2, 1, 128],
    ("c3", 3, [1, 1, 0, 128]),
    [3, 2, 1, 256],
    ("c3", 6, [1, 1, 0, 256]),
    [3, 2, 1, 512],
    ("c3", 9, [1, 1, 0, 512]),
    [3, 2, 1, 1024],
    ("c3", 3, [1, 1, 0, 1024]),
    "sppf",
]

class SpatialPyramidPoolingWithFixedBins(nn.Module):
    def __init__(self, in_channels, out_channel, pool_sizes=[[5, 1, 2], [5, 1, 2], [5, 1, 2]]):
        super(SpatialPyramidPoolingWithFixedBins, self).__init__()
        self.firstblock = ConvBNSiLUBlock(in_channels , out_channel=512, kernel_size=1, stride=1, padding=0)

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size[0], pool_size[1], pool_size[2]) for pool_size in pool_sizes])
        self.lastblock = ConvBNSiLUBlock(in_channels , out_channel= out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features=[]
        out=self.firstblock(x)
        out_copy= out
        for maxpool in self.maxpools:
            out=maxpool(out)
            features.append(out)

        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [out_copy], dim=1)
        # features = self.lastblock(features)
        return features

class ConvBNSiLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,bias=False):
        super(ConvBNSiLUBlock, self).__init__()
        #TODO change structure according to prediction part. if module in predict, bias is true and bn doesnt use.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x

class BottleNeck(nn.Module):
    def __init__(self, in_channels,):
        super(BottleNeck, self).__init__()
        reduced_channels = int(in_channels / 2)
        self.block0 = ConvBNSiLUBlock(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0)
        self.block1 = ConvBNSiLUBlock(reduced_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.block0(x)
        out = self.block1(out)
        out += residual
        return out

class BottleNeckCsp(nn.Module):
    def __init__(self, in_channels):
        super(BottleNeckCsp, self).__init__()
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
    def __init__(self, in_channels, out_channels, nblocks, kernel_size, stride, padding):
        super(C3, self).__init__()
        reduced_channels = int(in_channels / 2)
        self.block0= ConvBNSiLUBlock(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0)
        self.block1 = ConvBNSiLUBlock(in_channels, reduced_channels, kernel_size=3, stride=1, padding=1)
        cspblocks = []
        for i in range(nblocks):
            cspblocks.append(BottleNeckCsp(in_channels//2))
        self.cspblocks = nn.Sequential(*cspblocks)
        self.lastblock = ConvBNSiLUBlock(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        split0 = self.block0(x)
        split1 = self.block1(x)
        blocks = self.cspblocks(split0)
        out = torch.cat([split1, blocks], dim=1)
        out=self.lastblock(out)
        return out

class Yolov5(nn.Module):
    def __init__(self, config: list ,n_channels=3, num_classes=80):
        super(Yolov5, self).__init__()
        self.config=config
        self.n_channels=n_channels
        self.num_classes=num_classes
        self.layers=self._create_model()
    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            x = layer(x)
        return x
    def _create_model(self):
        blocks=[]
        in_channels=self.n_channels
        for module in self.config:
            if isinstance(module,list):
                kernel_size, stride, padding, out_channels = module
                blocks.append(ConvBNSiLUBlock(in_channels, out_channels, kernel_size, stride, padding))
                in_channels = out_channels
            if isinstance(module, tuple):
                type, nblocks, (kernel_size, stride, padding, out_channels) = module
                blocks.append(C3(in_channels, out_channels, nblocks, kernel_size, stride, padding))
                in_channels = out_channels
            if isinstance(module, str):
                out_channels=1024
                blocks.append(SpatialPyramidPoolingWithFixedBins(in_channels,out_channels))
                in_channels = out_channels

        return nn.Sequential(*blocks)


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 640
    # Usage example:

    Yv5 =Yolov5(config=config,num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = Yv5(x)
    # assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    # assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    # assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!",out.shape)
    #torch.split(x,x.size()[1]//2,dim=1)[0].size()