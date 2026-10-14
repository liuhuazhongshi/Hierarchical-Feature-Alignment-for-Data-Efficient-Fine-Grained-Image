#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CBAM（Convolutional Block Attention Module）模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAMBlock(nn.Module):
    """CBAM（Convolutional Block Attention Module）模块"""
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # 空间注意力
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = self.spatial_attention(torch.cat([spatial_avg, spatial_max], dim=1))
        x = x * spatial_att
        
        return x
