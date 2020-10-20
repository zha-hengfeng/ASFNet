import math
import torch
import torch.nn as nn
from functools import reduce
from .module.apf_att import APF_Moudle
from .module.da_att import CAM_Module
from .module.seg_head import Seg_Head
from .ResNet import *


class APFNet_CAM(nn.Module):
    def __init__(self, num_classes=19, backbone='res18', encoder_only=True, block_channel=32, only34=False, use3x3=False):
        super(APFNet_CAM, self).__init__()
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

        self.cam_1 = CAM_Module(in_dim=2 * block_channel)
        self.cam_2 = CAM_Module(in_dim=2 * block_channel)
        self.cam_3 = CAM_Module(in_dim=2 * block_channel)
        self.cam_4 = CAM_Module(in_dim=2 * block_channel)

        self.avgpool_1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_block_1 = nn.Sequential(nn.Conv2d(block_channel, 2 * block_channel, 3, 1, 1),
                                          nn.BatchNorm2d(2 * block_channel),
                                          nn.ReLU(inplace=True)
                                          )
        self.conv_block_3 = nn.Sequential(nn.Conv2d(4 * block_channel, 2 * block_channel, 3, 1, 1),
                                          nn.BatchNorm2d(2 * block_channel),
                                          nn.ReLU(inplace=True)
                                          )
        self.conv_block_4 = nn.Sequential(nn.Conv2d(256, 2 * block_channel, 3, 1, 1),
                                          nn.BatchNorm2d(2 * block_channel),
                                          nn.ReLU(inplace=True)
                                          )
        self.apf_m_pre = APF_Moudle(in_channels=2 * block_channel, out_channels=64, stride=1, M=2, r=16,
                                    L=num_classes)
        self.apf_m = APF_Moudle(in_channels=2 * block_channel, out_channels=64, stride=1, M=3, r=16,
                                L=num_classes)
        # self.classify = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.classify_backbone = Seg_Head(in_channel=256, num_classes=num_classes)
        self.classify_34 = Seg_Head(in_channel=2*block_channel, num_classes=num_classes)
        self.classify_final = Seg_Head(num_classes=num_classes)
        self.only34 = only34
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

        backbone_out = self.classify_backbone(block_4_out)
        backbone_out = torch.nn.functional.interpolate(backbone_out, input.size()[2:], mode='bilinear',
                                                align_corners=False)

        b3_out_conv = self.conv_block_3(block_3_out)
        b3_out_cam = self.cam_3(b3_out_conv)
        b4_out_conv = self.conv_block_4(block_4_out)
        b4_out_cam = self.cam_4(b4_out_conv)
        # block_34_out = self.apf_m_pre([b3_out_cam, b4_out_cam])

        # out34 = self.classify_34(block_34_out)
        # out34 = torch.nn.functional.interpolate(out34, input.size()[2:], mode='bilinear',
        #                                          align_corners=False)

        block_3_out_up = nn.functional.interpolate(b3_out_cam, block_2_out.size()[2:], mode='bilinear',
                                                    align_corners=False)
        block_4_out_up = nn.functional.interpolate(b4_out_cam, block_2_out.size()[2:], mode='bilinear',
                                                   align_corners=False)
        b1_out_down = self.avgpool_1(block_1_out)
        b1_out_down = self.conv_block_1(b1_out_down)
        b1_out_cam = self.cam_1(b1_out_down)
        b2_out_cam = self.cam_2(block_2_out)

        output = self.apf_m([b1_out_cam, b2_out_cam, block_4_out_up])

        output = self.classify_final(output)
        if not self.encoder_only:
            output = self.eac_module(output)
        output = torch.nn.functional.interpolate(output, input.size()[2:], mode='bilinear', align_corners=False)
        outputs = [output, backbone_out]

        return outputs