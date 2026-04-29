import torch
import torch.nn as nn
import torch.nn.functional as F


def remove_prefix(state_dict, prefix="module."):
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_dict[k[len(prefix):]] = v
        else:
            new_dict[k] = v
    return new_dict


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.PReLU(oup)
        )

    def forward(self, x):
        return self.conv(x)


class DepthWise(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1,
                       groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.PReLU(inp),

            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.PReLU(oup),
        )

    def forward(self, x):
        return self.conv(x)


class MiniFASNetV1SE(nn.Module):
    """ 80×80 version - 32 channels """
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(3, 32)
        self.conv2_dw = DepthWise(32, 32)
        self.conv3 = DepthWise(32, 64)
        self.conv4 = DepthWise(64, 128)
        self.conv5 = DepthWise(128, 128)

        self.fc = nn.Linear(128 * 5 * 5, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        x = F.max_pool2d(x, 2)

        x = self.conv5(x)
        x = F.avg_pool2d(x, x.size()[2:])

        x = x.view(x.size(0), -1)
        return self.fc(x)


class MiniFASNetV2(nn.Module):
    """ 80×80 version - 32 channels """
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(3, 32)
        self.conv2_dw = DepthWise(32, 64)
        self.conv3 = DepthWise(64, 128)
        self.conv4 = DepthWise(128, 128)

        self.fc = nn.Linear(128 * 5 * 5, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        x = F.avg_pool2d(x, x.size()[2:])

        x = x.view(x.size(0), -1)
        return self.fc(x)
