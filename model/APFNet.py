import math
import torch
import torch.nn as nn
from functools import reduce
from .module.apf_att import APF_Moudle
from .module.da_att import CAM_Module
from .ResNet import *


class APFNet(nn.Module):
    def __init__(self, num_classes=19, backbone='res18', encoder_only=True, block_channel=32, use3x3=False):
        super(APFNet, self).__init__()
        if backbone == 'res18':
            backbone = ResNet18(pretrained=False, block_channel=block_channel)
        if backbone == 'res34':
            backbone = ResNet34(pretrained=False, block_channel=block_channel)
        self.conv1 = backbone.conv1
        if use3x3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3,  block_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(block_channel//2, block_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.Conv2d(block_channel, block_channel, kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.bn1 = backbone.bn1
        self.relu1 = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.conv_block_4 = nn.Sequential(nn.Conv2d(256, 2*block_channel, 3, 1, 1),
                                          nn.BatchNorm2d(2*block_channel),
                                          nn.ReLU(inplace=True)
                                          )

        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_block_1 = nn.Sequential(nn.Conv2d(block_channel, 2 * block_channel, 3, 1, 1),
                                          nn.BatchNorm2d(2 * block_channel),
                                          nn.ReLU(inplace=True)
                                          )
        self.apf_m = APF_Moudle(in_channels=2*block_channel, out_channels=64, stride=1, M=3, r=16,
                                      L=num_classes)
        # self.classify = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.classify_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.encoder_only = encoder_only
        # if not encoder_only:
        #     self.eac_module = EAC_Module(num_classes, num_classes)


    def forward(self, input):
        conv_first = self.conv1(input)          # block_c
        conv_first = self.bn1(conv_first)
        conv_first = self.relu1(conv_first)
        block_1_out = self.layer1(conv_first)   # block_c
        block_2_out = self.layer2(block_1_out)  # 2 * block_c
        block_3_out = self.layer3(block_2_out)  # 4 * block_c
        block_4_out = self.layer4(block_3_out)  # 256
        # output_3 = self.classify_3(block_3_out)
        # output_4 = self.classify_4(block_4_out)
        # output = self.sk_module([output_3, output_4])

        block_4_out_up = self.conv_block_4(block_4_out)
        block_4_out_up = nn.functional.interpolate(block_4_out_up, block_2_out.size()[2:], mode='bilinear', align_corners=False)
        # output = self.sk_module([block_2_out, block_4_out_up])

        block_1_out_down = self.avgpool(block_1_out)
        block_1_out_down = self.conv_block_1(block_1_out_down)
        output = self.apf_m([block_1_out_down, block_2_out, block_4_out_up])

        output = self.classify_out(output)
        if not self.encoder_only:
            output = self.eac_module(output)
        output = torch.nn.functional.interpolate(output, input.size()[2:], mode='bilinear', align_corners=False)
        return output
