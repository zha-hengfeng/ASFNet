import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models
from torchvision.models.resnet import model_urls
# from torchvision.models.segmentation.segmentation import model_urls


class FCN_ResNet_18(nn.Module):
    def __init__(self, num_classes=19, encoder_only=False, block_channel=32, use3x3=False):
        super(FCN_ResNet_18, self).__init__()
        backbone = ResNet18(pretrained=False, block_channel=block_channel)
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

        self.classify = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
        )
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
        return output


class EAC_Module(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EAC_Module, self).__init__()
        self.chanel_in = in_dim
        self.max_pool_1 = nn.MaxPool2d(2, stride=2)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        pool = self.max_pool_1(x)
        pool_batch, pool_C, pool_h, pool_w = pool.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(pool).view(m_batchsize, -1, pool_h*pool_w)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(pool).view(m_batchsize, -1, pool_h*pool_w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, block_channel, num_classes=1000):
        self.inplanes = block_channel
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, block_channel, kernel_size=7, stride=2, padding=3, bias=False)   # 第一次下采样
        # add by @zhA
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=False),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, bias=False),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=3, bias=False),
        # )
        self.bn1 = nn.BatchNorm2d(block_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # 第二次下采样
        self.layer1 = self._make_layer(block, block_channel, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 2*block_channel, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4*block_channel, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18(pretrained=False, block_channel=64, **kwargs):
    # trained(bool): If True, returns a model pre - trained on ImageNet
    model = ResNet(BasicBlock, [2, 2, 2, 2], block_channel, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model.load_state_dict(torch.load("model/model_zoo/resnet18-5c106cde.pth"))
    return model


def ResNet34(pretrained=False, block_channel=32, **kwargs):
    # trained(bool): If True, returns a model pre - trained on ImageNet
    model = ResNet(BasicBlock, [3, 4, 6, 3], block_channel, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model.load_state_dict(torch.load("model/model_zoo/resnet18-5c106cde.pth"))
    return model