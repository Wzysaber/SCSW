import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict


class StdConv2d(nn.Conv2d):
    # StdConv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3))
    def forward(self, x):
        w = self.weight  # [64, 3, 7, 7] 64 ge channel 3 7x7
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        # x = x.transpose([0, 3, 1, 2])
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = StdConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes, eps=1e-6)

        self.conv2 = StdConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, planes, eps=1e-6)

        self.conv3 = StdConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, planes * 4, eps=1e-6)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # if use_amm == True:
        #     self.amm = AMM(planes * 4, 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3])

        self.inplanes = 1024

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, use_amm=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion, eps=1e-6),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model_name = "resnet50"

    if pretrained:
        state_dict = torch.load(
            "/home0/students/master/2022/wangzy/Pycharm-Remote(161)/FSSS_Model/PTH/resnet50-19c8e357.pth")
        model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == '__main__':
    model = resnet50()
    model.eval()
    image = torch.randn(2, 3, 256, 256)

    output = model(image)
    print("input:", image.shape)
    print("output:", output.shape)
