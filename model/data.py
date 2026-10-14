#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理模块
"""
import math
import pickle
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import RandomErasing, AutoAugment, AutoAugmentPolicy
from torchvision import datasets, transforms

# CIFAR-100 均值和标准差
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# 获取细粒度到粗粒度的映射
train_path = Path("./data/cifar-100-python/train")
with train_path.open("rb") as f:
    train_data = pickle.load(f, encoding="bytes")

# 获取细粒度和粗粒度标签
fine_labels = train_data[b'fine_labels']
coarse_labels = train_data[b'coarse_labels']

# 创建细粒度到粗粒度的映射
fine2coarse_id = torch.zeros(100, dtype=torch.long)
for fine, coarse in zip(fine_labels, coarse_labels):
    fine2coarse_id[fine] = coarse

# CutMix参数
cutmix_beta = 1.0

def cutmix_collate(batch):
    """返回 CutMix 后的 tensor + 两个 target + lam"""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    # 采样beta分布的lam值
    lam = torch.distributions.Beta(cutmix_beta, cutmix_beta).sample().item()
    bsz = images.size(0)
    index = torch.randperm(bsz)
    W, H = images.size(-1), images.size(-2)
    
    # 根据lam值计算裁剪区域大小，确保至少为1x1
    cut_rat = math.sqrt(1. - lam)
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))
    
    # 随机选择裁剪区域的左上角坐标，确保裁剪区域有效
    bbx1 = torch.randint(0, W - cut_w + 1, (1,)).item()
    bby1 = torch.randint(0, H - cut_h + 1, (1,)).item()
    bbx2 = bbx1 + cut_w
    bby2 = bby1 + cut_h
    
    # 执行CutMix操作
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    
    # 计算实际的lam值
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)
    
    return images, labels, labels[index], lam
