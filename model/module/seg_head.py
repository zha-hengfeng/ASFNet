import torch
import torch.nn as nn


class Seg_Head(nn.Module):
    def __init__(self, in_channel=64, num_classes=19):
        super(Seg_Head, self).__init__()
        self.classify = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, input):
        output = self.classify(input)

        return output
