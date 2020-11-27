import math
import torch
import torch.nn as nn
from functools import reduce
from .module.apf_att import APF_Moudle
from .module.da_att import CAM_Module
from .module.alignmodule import AlignModule, AlignModule_Segmap
from .ResNet import *


class APFNetv2_sf(nn.Module):
    def __init__(self, num_classes=19, backbone='res18', encoder_only=True, block_channel=32, use3x3=False):
        super(APFNetv2_sf, self).__init__()
        if backbone == 'res18':
            backbone = ResNet18(pretrained=False, block_channel=block_channel)
        if backbone == 'res34':
            backbone = ResNet34(pretrained=False, block_channel=block_channel)
        self.conv1 = backbone.conv1
        if use3x3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, block_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(block_channel // 2, block_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.Conv2d(block_channel, block_channel, kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.bn1 = backbone.bn1
        self.relu1 = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.classify_4 = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.classify_3 = nn.Conv2d(4 * block_channel, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.classify_2 = nn.Conv2d(2 * block_channel, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.b4_b2 = AlignModule(num_classes, num_classes)
        self.b3_b2 = AlignModule(num_classes, num_classes)

        self.apf_m = APF_Moudle(in_channels=num_classes, out_channels=num_classes, stride=1, M=3, r=16,
                                L=num_classes)

        self.classify_out = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.classify_out = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=1, bias=False),
        # )

        self.encoder_only = encoder_only
        # if not encoder_only:
        #     self.eac_module = EAC_Module(num_classes, num_classes)

    def forward(self, input):
        conv_first = self.conv1(input)  # block_c
        conv_first = self.bn1(conv_first)
        conv_first = self.relu1(conv_first)
        block_1_out = self.layer1(conv_first)  # block_c
        block_2_out = self.layer2(block_1_out)  # 2 * block_c
        block_3_out = self.layer3(block_2_out)  # 4 * block_c
        block_4_out = self.layer4(block_3_out)  # 256
        output_2 = self.classify_2(block_2_out)
        output_3 = self.classify_3(block_3_out)
        output_4 = self.classify_4(block_4_out)
        # output = self.sk_module([output_3, output_4])
        out_4_up = self.b4_b2([output_2, output_4])
        out_3_up = self.b4_b2([output_2, output_3])

        output = self.apf_m([output_2, out_3_up, out_4_up])

        output = self.classify_out(output)
        if not self.encoder_only:
            output = self.eac_module(output)
        output = torch.nn.functional.interpolate(output, input.size()[2:], mode='bilinear', align_corners=False)
        return [output]


class APFNetv2_sf_2(nn.Module):
    def __init__(self, num_classes=19, backbone='res18', encoder_only=True, block_channel=32, use3x3=False):
        super(APFNetv2_sf_2, self).__init__()
        if backbone == 'res18':
            backbone = ResNet18(pretrained=False, block_channel=block_channel)
        if backbone == 'res34':
            backbone = ResNet34(pretrained=False, block_channel=block_channel)
        self.conv1 = backbone.conv1
        if use3x3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, block_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(block_channel // 2, block_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.Conv2d(block_channel, block_channel, kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.bn1 = backbone.bn1
        self.relu1 = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # self.classify_4 = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.classify_3 = nn.Conv2d(4 * block_channel, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.classify_2 = nn.Conv2d(2 * block_channel, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.b4_b2 = AlignModule_Segmap(2 * block_channel, 256, 2 * block_channel)
        self.b3_b2 = AlignModule_Segmap(2 * block_channel, 4 * block_channel, 2 * block_channel)

        self.apf_m = APF_Moudle(in_channels=2 * block_channel, out_channels=2 * block_channel, stride=1, M=3, r=16,
                                L=num_classes)

        self.classify_out = nn.Conv2d(2 * block_channel, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.classify_out = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=1, bias=False),
        # )

        self.encoder_only = encoder_only
        # if not encoder_only:
        #     self.eac_module = EAC_Module(num_classes, num_classes)

    def forward(self, input):
        conv_first = self.conv1(input)  # block_c
        conv_first = self.bn1(conv_first)
        conv_first = self.relu1(conv_first)
        block_1_out = self.layer1(conv_first)  # block_c
        block_2_out = self.layer2(block_1_out)  # 2 * block_c
        block_3_out = self.layer3(block_2_out)  # 4 * block_c
        block_4_out = self.layer4(block_3_out)  # 256
        output_2 = self.classify_2(block_2_out)
        # output_3 = self.classify_3(block_3_out)
        # output_4 = self.classify_4(block_4_out)
        # output = self.sk_module([output_3, output_4])
        out_4_up = self.b4_b2([block_2_out, block_4_out])
        out_3_up = self.b3_b2([block_2_out, block_3_out])

        output = self.apf_m([block_2_out, out_3_up, out_4_up])

        output = self.classify_out(output)
        if not self.encoder_only:
            output = self.eac_module(output)
        output = torch.nn.functional.interpolate(output, input.size()[2:], mode='bilinear', align_corners=False)
        return [output]