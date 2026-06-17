import torch
import torch.nn as nn
import sys
# ConvAdapter(inplanes, planes, 
#             kernel_size=3, 
#             padding=1,
#             width=inplanes // tuning_config['adapt_size'], 
#             stride=stride, 
#             groups=inplanes // tuning_config['adapt_size'], 
#             dilation=1,
#             act_layer=nn.ReLU)
class ConvAdapter(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rank, alpha, 
                kernel_size=3, padding=1, stride=1, groups=1, dilation=1, act_layer=None):
        super().__init__()

        if act_layer is None:
            act_layer = nn.Identity()

        # depth-wise conv
        self.conv1 = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, dilation=dilation)

        self.act = act_layer

        # poise-wise conv
        self.conv2 = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)
        self.alpha = alpha

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out * self.alpha

        return out
    

class ConvUpscaleAdapter(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rank, alpha, 
                kernel_size=3, padding=1, stride=1, groups=1, dilation=1, act_layer=None):
        super().__init__()

        if act_layer is None:
            act_layer = nn.Identity()

        # depth-wise conv
        self.conv1 = nn.Conv2d(in_channels, rank * 4, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, dilation=dilation)

        self.act = act_layer
        self.upscale = nn.PixelShuffle(2)

        # poise-wise conv
        self.conv2 = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)
        self.alpha = alpha

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.upscale(out)
        out = self.conv2(out)
        out = out * self.alpha

        return out


class SubpelConvWithAdapter(torch.nn.Module):
    def __init__(self, subpel_conv, rank, alpha, activation = nn.Identity, divide_rank = False) -> None:
        super().__init__()
        assert isinstance(subpel_conv[0], nn.Conv2d)

        if divide_rank:
            rank = subpel_conv[0].in_channels // rank
        self.subpel_conv = subpel_conv
        self.adapter = ConvUpscaleAdapter(
            in_channels=subpel_conv[0].in_channels, 
            out_channels=subpel_conv[0].out_channels // 4,
            rank= rank,
            alpha=alpha,
            kernel_size=subpel_conv[0].kernel_size,
            padding=subpel_conv[0].padding,
            stride=subpel_conv[0].stride,
            groups= rank,
            dilation=subpel_conv[0].dilation,
            act_layer=activation)

    def forward(self, x):
        return self.subpel_conv(x) + self.adapter(x)
    

class ConvWithAdapter(torch.nn.Module):
    def __init__(self, conv:nn.Conv2d, rank, alpha, activation = nn.Identity, divide_rank = False):
        super().__init__()
        self.conv = conv

        if divide_rank:
            rank = conv.in_channels // rank
        
        rank = max(rank, 1)
        group_rank = rank

        if conv.in_channels % group_rank != 0:
            group_rank = 1

        # if rank > conv.in_channels:
        #     group_rank = 1
        
        # print(conv.in_channels, group_rank)

        self.adapter = ConvAdapter(
            in_channels=conv.in_channels, 
            out_channels=conv.out_channels,
            rank= rank,
            alpha=alpha,
            kernel_size=conv.kernel_size,
            padding=conv.padding,
            stride=conv.stride,
            groups=group_rank, 
            dilation=conv.dilation,
            act_layer=activation)
        
    def forward(self, x):
        return self.conv(x) + self.adapter(x)