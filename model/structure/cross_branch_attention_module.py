#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨分支注意力模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_branch_attention import CrossBranchAttention

class CrossBranchAttentionModule(nn.Module):
    """跨分支注意力模块"""
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.attention = CrossBranchAttention(embed_dim, dropout)
        # 移除额外的线性层和层归一化，因为注意力模块内部已经有了
        # 简化设计以提高稳定性
    
    def forward(self, query, key_value):
        out = self.attention(query, key_value)
        return out
