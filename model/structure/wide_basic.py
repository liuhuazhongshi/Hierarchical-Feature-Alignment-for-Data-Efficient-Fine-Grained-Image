#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WideResNet基础模块
"""
import torch.nn as nn
import torch.nn.functional as F

class WideBasic(nn.Module):
    """WideResNet 基础模块"""
    def __init__(self, c_in, c_out, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride, 1, bias=False)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Conv2d(c_in, c_out, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.drop(out)
        out = self.conv2(out)
        return out + self.shortcut(x)
