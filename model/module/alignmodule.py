import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature= x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        # h_feature = F.grid_sample(h_feature_orign, flow.permute(0, 2, 3, 1))
        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # print(norm.shape)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        # print(w.shape)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        # print(h.shape)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        # print(grid.shape)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        # grid = grid + flow.permute(0, 2, 3, 1)

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignModule_Segmap(nn.Module):
    def __init__(self, in_l, in_h, outplane):
        super(AlignModule_Segmap, self).__init__()
        self.down_h = nn.Conv2d(in_h, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(in_l, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class AlignModule_GAU(nn.Module):
    def __init__(self, in_low, out_channel):
        super(AlignModule_GAU, self).__init__()
        self.low_conv = nn.Conv2d(in_low, out_channel, kernel_size=3, padding=1, bias=False)
        self.high_conv = nn.Conv2d(in_low, out_channel, 1, bias=False )
        self.atten = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        low_feature, h_feature = x
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.low_conv(low_feature)
        h_feature = self.high_conv(h_feature)
        h_feature = self.relu(self.bn(h_feature))
        h_atten = self.atten(h_feature)
        low_feature = h_atten * low_feature
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        output = low_feature + h_feature

        return output


class AlignModule_GAU_2(nn.Module):
    def __init__(self, in_low, in_high, out_channel):
        super(AlignModule_GAU, self).__init__()
        self.low_conv = nn.Conv2d(in_low, out_channel, kernel_size=3, padding=1, bias=False)
        self.high_conv = nn.Conv2d(in_high, out_channel, 1, bias=False )
        self.atten = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        low_feature, h_feature = x
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.low_conv(low_feature)
        h_feature = F.relu(F.batch_norm(self.high_conv(h_feature)))
        h_atten = self.atten(h_feature)
        low_feature = low_feature * h_atten
        h_feature = F.interpolate(h_feature, size, mode="bilinear", align_corners=False)
        output = low_feature + h_feature

        return output


if __name__ == '__main__':
    module_test = AlignModule(19, 19)
    input_size_1 = (2, 19, 64, 128)
    input_size_2 = (2, 19, 128, 256)
    input_1 = torch.randn(*input_size_1, device='cpu')
    input_2 = torch.randn(*input_size_2, device='cpu')

    module_test([input_1, input_2])