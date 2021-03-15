import math
import torch
import torch.nn as nn
from model.module.eac_att import EAC_Module
from model.ResNet import *


class FCN_ResNet(nn.Module):
    def __init__(self, num_classes=19, encoder_only=True, backbone="res34", block_channel=32, use3x3=False):
        super(FCN_ResNet, self).__init__()
        # build backbone
        if backbone == 'res18':
            backbone = ResNet18(pretrained=False, block_channel=block_channel)
        if backbone == 'res34':
            backbone = ResNet34(pretrained=False, block_channel=block_channel)

        self.conv1 = backbone.conv1
        if use3x3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3,  block_channel//2, kernel_size=5, stride=1, padding=1, bias=False),
                nn.Conv2d(block_channel//2, block_channel, kernel_size=3, stride=2, padding=1, bias=False),
                # nn.Conv2d(block_channel, block_channel, kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.bn1 = backbone.bn1
        self.relu1 = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.classify = nn.Sequential(
            nn.Conv2d(8 * block_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # self.classify = nn.Sequential(
        #     nn.Conv2d(8 * block_channel, 8 * block_channel, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(8 * block_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8 * block_channel, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
        # )
        # if 8 * block_channel >= 256:
        #     self.classify = nn.Sequential(
        #         nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
        #     )

        self.encoder_only = encoder_only
        if not encoder_only:
            self.eac_module = EAC_Module(num_classes, num_classes)

    def forward(self, input):
        conv_first = self.conv1(input)
        conv_first = self.bn1(conv_first)
        conv_first = self.relu1(conv_first)
        block_1_out = self.layer1(conv_first)
        block_2_out = self.layer2(block_1_out)
        block_3_out = self.layer3(block_2_out)
        block_4_out = self.layer4(block_3_out)
        output = self.classify(block_4_out)
        if not self.encoder_only:
            output = self.eac_module(output)
        output = torch.nn.functional.interpolate(output, input.size()[2:], mode='bilinear', align_corners=False)
        return [output]
