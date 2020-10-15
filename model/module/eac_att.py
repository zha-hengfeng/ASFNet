import torch
import torch.nn as nn


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
