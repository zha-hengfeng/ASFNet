import torch
import torch.nn as nn
from functools import reduce


class APF_Moudle(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(APF_Moudle, self).__init__()
        # d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.conv_1 = nn.Sequential(
            nn.Conv2d(out_channels*M, out_channels, 1, 1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Sigmoid()
                                 )
        self.fc2 = nn.Conv2d(out_channels, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        batch_size = outputs[0].size(0)
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())

            outputs[i] = conv(outputs[i])
        # U = reduce(lambda x, y: x + y, outputs)
        U = torch.cat(outputs, dim=1)
        U = self.conv_1(U)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, outputs, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V