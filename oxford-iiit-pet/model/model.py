#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import fine2coarse_id

# 导入结构模块
from .structure.wide_basic import WideBasic
from .structure.cross_branch_attention import CrossBranchAttention
from .structure.cross_branch_attention_module import CrossBranchAttentionModule
from .structure.se_block import SEBlock
from .structure.cbam_block import CBAMBlock



class WideResNet(nn.Module):
    """WideResNet模型，支持层次化解耦"""
    def __init__(self, depth=16, widen=8, nc_coarse=8, nc_fine=37, dropout=0.0, use_coarse_branch=True, use_cross_attention=True, use_kl=True, attention_type='cross_branch'):
        super().__init__()
        # 添加模型深度合法性检查
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must be of the form 6n+4 (e.g., 28, 40)")
            
        n = (depth - 4) // 6
        k = widen
        nStages = [16, 16*k, 32*k, 64*k]
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(WideBasic, nStages[1], n, 1, dropout)
        self.layer2 = self._make_layer(WideBasic, nStages[2], n, 2, dropout)
        self.layer3 = self._make_layer(WideBasic, nStages[3], n, 2, dropout)
        self.bn = nn.BatchNorm2d(nStages[3])
        
        # 保存细粒度到粗粒度的映射关系
        self.use_coarse_branch = use_coarse_branch
        self.use_cross_attention = use_cross_attention
        self.use_kl = use_kl
        self.attention_type = attention_type  # 新增注意力类型参数
        
        # 特征提取层
        if use_coarse_branch:
            # 超类分支：使用更大的卷积核和更少的非线性变换，专注于全局特征
            # 将SE/CBAM集成到分支内部
            super_branch_layers = [
                nn.Conv2d(nStages[3], nStages[3], 5, 1, 2, bias=False),  # 使用5x5卷积核
                nn.BatchNorm2d(nStages[3]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            
            # 如果使用SE或CBAM注意力，将其集成到分支内部
            if use_cross_attention and attention_type == 'se':
                super_branch_layers.append(SEBlock(nStages[3]))
            elif use_cross_attention and attention_type == 'cbam':
                super_branch_layers.append(CBAMBlock(nStages[3]))
            
            super_branch_layers.extend([
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ])
            
            self.super_branch = nn.Sequential(*super_branch_layers)
        
        # 细粒度分支：使用更小的卷积核和更多的非线性变换，专注于细节特征
        # 将SE/CBAM集成到分支内部
        fine_branch_layers = [
            nn.Conv2d(nStages[3], nStages[3], 1, 1, 0, bias=False),
            nn.BatchNorm2d(nStages[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 使用统一的dropout率
            nn.Conv2d(nStages[3], nStages[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(nStages[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)  # 使用统一的dropout率
        ]
        
        # 如果使用SE或CBAM注意力，将其集成到分支内部
        if use_cross_attention and attention_type == 'se':
            fine_branch_layers.append(SEBlock(nStages[3]))
        elif use_cross_attention and attention_type == 'cbam':
            fine_branch_layers.append(CBAMBlock(nStages[3]))
        
        fine_branch_layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        ])
        
        self.fine_branch = nn.Sequential(*fine_branch_layers)
        
        if use_cross_attention:
            if attention_type == 'cross_branch':
                # 添加分支间的交互机制 - 使用跨分支注意力机制
                self.attention_super = CrossBranchAttentionModule(nStages[3], dropout=dropout)
                self.attention_fine = CrossBranchAttentionModule(nStages[3], dropout=dropout)
            elif attention_type in ['se', 'cbam']:
                # SE或CBAM注意力将在分支内部应用，不需要额外的模块
                self.attention_super = None
                self.attention_fine = None
            elif attention_type == 'none':
                # 无注意力
                self.attention_super = None
                self.attention_fine = None
        else:
            # 简单的线性注意力（用于消融实验）- 明确告诉用户这是线性自注意力
            self.attention_super = nn.Sequential(
                nn.Linear(nStages[3], nStages[3] // 4),
                nn.ReLU(inplace=True),
                nn.Linear(nStages[3] // 4, nStages[3]),
                nn.Sigmoid()
            )
            
            self.attention_fine = nn.Sequential(
                nn.Linear(nStages[3], nStages[3] // 4),
                nn.ReLU(inplace=True),
                nn.Linear(nStages[3] // 4, nStages[3]),
                nn.Sigmoid()
            )

        # 分类器
        if use_coarse_branch:
            self.super_cls = nn.Linear(nStages[3], nc_coarse)
        
        # 细粒度分类器
        self.fine_cls = nn.Linear(nStages[3], nc_fine)

        # 注册细粒度到粗粒度的映射关系
        self.register_buffer("f2c", fine2coarse_id.detach().clone())
        # 注册映射矩阵作为buffer，避免每次forward重新创建
        if use_coarse_branch:
            self.register_buffer("_map_mat", 
                torch.zeros(nc_fine, nc_coarse).scatter_(
                    1, self.f2c.unsqueeze(1), 1.0))

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        """创建WideResNet层"""
        layers = [block(self.in_planes, planes, stride, dropout)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))



        # 分支分离的特征提取
        if self.use_coarse_branch:
            # 超类分支：特征提取已包含SE/CBAM注意力（如果启用）
            fs_raw = self.super_branch(out)
        
        # 细粒度分支：特征提取已包含SE/CBAM注意力（如果启用）
        ff_raw = self.fine_branch(out)
        

        
        if self.use_coarse_branch and self.use_cross_attention and self.attention_super is not None:
            # 分支间的注意力交互 - 改进的跨分支注意力
            # 超类分支使用细粒度分支的特征作为引导（query来自超类，key/value来自细粒度）
            fs_attended = self.attention_super(fs_raw, ff_raw)
            
            # 细粒度分支使用超类分支的特征作为引导（query来自细粒度，key/value来自超类）
            ff_attended = self.attention_fine(ff_raw, fs_raw)
        elif self.use_coarse_branch:
            # 无注意力交互
            fs_attended = fs_raw
            ff_attended = ff_raw
        else:
            # 无粗粒度分支
            ff_attended = ff_raw
        

        

        

        

        
        # 生成最终的分类logits
        if self.use_coarse_branch:
            logit_s = self.super_cls(fs_attended)
        else:
            logit_s = None
        
        # 计算分类logits
        logit_f = self.fine_cls(ff_attended)
        
        if self.use_coarse_branch:
            return logit_s, logit_f, fs_attended, ff_attended
        else:
            return logit_f

    def hier_kl(self, logit_f, logit_c, lam):
        """层次KL散度损失"""
        prob_f = F.softmax(logit_f, 1)
        # 使用预注册的映射矩阵将细粒度概率映射到粗粒度概率
        prob_c = prob_f @ self._map_mat.to(prob_f.device)
        
        # 计算KL散度：粗粒度预测分布与细粒度映射到粗粒度的分布之间的差异
        # 注意：kl_div的第一个参数是log_softmax(预测), 第二个参数是目标分布
        return lam * F.kl_div(F.log_softmax(logit_c, 1), prob_c, reduction='batchmean')

    def orth_loss(self, fs, ff, lam):
        """正交正则化：两个分支特征的L2内积的绝对值 + Gram矩阵正则化"""
        # 基础正交损失
        fs = F.normalize(fs, p=2, dim=1)
        ff = F.normalize(ff, p=2, dim=1)
        loss_inter = torch.abs((fs * ff).sum(1)).mean()
        
        # 增加Gram矩阵正则化，使同一分支内的特征也保持正交
        # 将特征reshape为batch_size × feature_dim
        batch_size = fs.size(0)
        fs_reshaped = fs.view(batch_size, -1)
        ff_reshaped = ff.view(batch_size, -1)
        
        # 计算Gram矩阵（batch_size × feature_dim × feature_dim）
        gram_s = torch.bmm(fs_reshaped.unsqueeze(2), fs_reshaped.unsqueeze(1))
        gram_f = torch.bmm(ff_reshaped.unsqueeze(2), ff_reshaped.unsqueeze(1))
        
        # 目标Gram矩阵：对角线为1，非对角线为0（完全正交）
        target = torch.eye(fs_reshaped.size(1), device=fs.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 计算Gram矩阵与目标矩阵的平方损失
        loss_gram_s = ((gram_s - target) ** 2).mean()
        loss_gram_f = ((gram_f - target) ** 2).mean()
        
        # 综合所有损失
        return lam * (loss_inter + loss_gram_s + loss_gram_f)
