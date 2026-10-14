#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨分支注意力机制
"""
import torch.nn as nn
import torch

class CrossBranchAttention(nn.Module):
    """跨分支注意力机制"""
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        
        # 线性变换层 - 查询来自当前分支，键值来自另一个分支
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 层归一化
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_key_value = nn.LayerNorm(embed_dim)
        self.norm_output = nn.LayerNorm(embed_dim)  # 输出层归一化
        
        # 残差连接（直接使用Identity）
        self.residual = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key_value):
        # query: [batch_size, embed_dim] - 当前分支的特征
        # key_value: [batch_size, embed_dim] - 另一个分支的特征
        
        # 保存原始query用于残差连接
        original_query = query
        
        # 输入归一化
        query = self.norm_query(query)
        key_value = self.norm_key_value(key_value)
        
        # 线性变换
        q = self.q_proj(query).unsqueeze(1)  # [batch_size, 1, embed_dim]
        k = self.k_proj(key_value).unsqueeze(2)  # [batch_size, embed_dim, 1]
        v = self.v_proj(key_value).unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 计算注意力分数 - 点积注意力
        # [batch_size, 1, embed_dim] x [batch_size, embed_dim, 1] -> [batch_size, 1, 1]
        attn_scores = torch.bmm(q, k) * self.scale
        attn_weights = self.softmax(attn_scores)  # [batch_size, 1, 1]
        
        # 应用注意力权重到值向量
        attn_output = torch.bmm(attn_weights, v).squeeze(1)  # [batch_size, embed_dim]
        attn_output = self.dropout(attn_output)  # 使用dropout防止过拟合
        
        # 最终线性变换和残差连接
        output = self.out_proj(attn_output)
        output = output + self.residual(original_query)  # 残差连接到原始输入
        output = self.norm_output(output)  # 输出层归一化
        
        return output
