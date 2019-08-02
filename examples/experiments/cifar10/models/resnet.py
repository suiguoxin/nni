'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel_size, kernel_size, pooling_size, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = channel_size

        self.conv1 = nn.Conv2d(3, out_channels=channel_size,
                               kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.pooling_size = pooling_size

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        print('0: ', x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        print('1: ', out.size())
        out = self.layer1(out)
        print('2: ', out.size())
        out = self.layer2(out)
        print('3: ', out.size())
        out = self.layer3(out)
        print('4: ', out.size())
        out = self.layer4(out)
        print('5: ', out.size())
        out = F.avg_pool2d(out, self.pooling_size)  # TODO
        print('6: ', out.size())
        out = out.view(out.size(0), -1)
        print('7: ', out.size())
        out = self.linear(out)
        print('8: ', out.size())
        return out


def ResNet18(channel_size=128, kernel_size=3, pooling_size=4):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel_size, kernel_size, pooling_size)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


test()
