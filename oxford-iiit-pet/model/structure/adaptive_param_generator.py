#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自适应参数生成模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveParamGenerator(nn.Module):
    """
    自适应参数生成模块：根据输入特征动态生成网络参数
    - 动态生成卷积核或线性层参数
    - 使模型能够自适应不同类型的输入
    - 提高模型的灵活性和表达能力
    """
    def __init__(self, feature_dim, param_dim, num_params, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.param_dim = param_dim
        self.num_params = num_params
        
        # 参数生成器网络
        self.param_generator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_params * param_dim)
        )
        
        # 注意力机制，用于选择重要特征
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # 可学习的参数基
        self.param_base = nn.Parameter(torch.randn(num_params, param_dim))
        
    def forward(self, x):
        """
        参数：
            x: [batch_size, feature_dim] - 输入特征
        返回：
            generated_params: [batch_size, num_params, param_dim] - 生成的参数
            param_attention: [batch_size, feature_dim] - 特征注意力权重
        """
        batch_size = x.size(0)
        
        # 学习特征注意力权重
        param_attention = self.feature_attention(x)
        x_attended = x * param_attention
        
        # 生成参数
        generated_params = self.param_generator(x_attended)
        generated_params = generated_params.view(batch_size, self.num_params, self.param_dim)
        
        # 添加参数基
        generated_params = generated_params + self.param_base.unsqueeze(0)
        
        return generated_params, param_attention

class AdaptiveConvolution(nn.Module):
    """
    自适应卷积层：使用动态生成的卷积核
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, feature_dim=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 计算卷积核参数数量
        kernel_params = in_channels * out_channels * kernel_size * kernel_size
        bias_params = out_channels
        
        # 创建参数生成器
        self.kernel_generator = AdaptiveParamGenerator(
            feature_dim=feature_dim,
            param_dim=kernel_params,
            num_params=1
        )
        
        self.bias_generator = AdaptiveParamGenerator(
            feature_dim=feature_dim,
            param_dim=bias_params,
            num_params=1
        )
        
        # 平均池化用于生成全局特征
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x, features):
        """
        参数：
            x: [batch_size, in_channels, height, width] - 输入特征图
            features: [batch_size, feature_dim] - 用于生成参数的特征
        返回：
            out: [batch_size, out_channels, height, width] - 卷积输出
        """
        batch_size = x.size(0)
        
        # 生成卷积核参数
        kernel_params, _ = self.kernel_generator(features)
        kernel_params = kernel_params.view(batch_size, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        # 生成偏置参数
        bias_params, _ = self.bias_generator(features)
        bias_params = bias_params.view(batch_size, self.out_channels)
        
        # 使用分组卷积实现批量处理（每个样本一个分组）
        # 重排输入以适应分组卷积
        x_reshaped = x.view(1, batch_size * self.in_channels, x.size(2), x.size(3))
        
        # 重排卷积核以适应分组卷积
        kernel_reshaped = kernel_params.view(batch_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        # 重排偏置以适应分组卷积
        bias_reshaped = bias_params.view(batch_size * self.out_channels)
        
        # 应用分组卷积
        out = F.conv2d(
            x_reshaped,
            kernel_reshaped,
            bias=bias_reshaped,
            stride=self.stride,
            padding=self.padding,
            groups=batch_size
        )
        
        # 恢复输出形状
        out = out.view(batch_size, self.out_channels, out.size(2), out.size(3))
        
        return out
