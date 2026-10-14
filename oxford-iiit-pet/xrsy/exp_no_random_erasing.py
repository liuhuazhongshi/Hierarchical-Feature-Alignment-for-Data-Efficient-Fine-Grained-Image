#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
消融实验：无RandomErasing数据增强（No Random Erasing）
======================================================================

移除RandomErasing数据增强的模型：
- 双分支结构（超类分支 + 细粒度分支）
- 保留层次一致性损失（KL）
- 保留正交损失
- 保留CutMix数据增强
- 保留SAM优化器
- 去掉RandomErasing数据增强

Author : Like Li
Date   : 2025-12-05
======================================================================
"""
import sys
import os
sys.path.insert(0, os.path.abspath("./"))
from model.experiment import BaseExperiment
from model.args import parse_args
from torchvision.transforms import RandAugment

class NoRandomErasingExperiment(BaseExperiment):
    def get_data_loader(self, batch_size, use_cutmix=True):
        """获取数据加载器，不使用RandomErasing数据增强"""
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
            # 去掉RandomErasing(p=0.5, scale=(0.02, 0.15))
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
        
        train_set = datasets.CIFAR100(root="./data", train=True, download=False, transform=train_tf)
        test_set  = datasets.CIFAR100(root="./data", train=False, download=False, transform=test_tf)
        
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=6, 
            pin_memory=True, 
            drop_last=True,
            collate_fn=cutmix_collate if use_cutmix else None
        )
        
        test_loader = DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=6, 
            pin_memory=True
        )
        
        return train_loader, test_loader

if __name__ == "__main__":
    args = parse_args()
    experiment = NoRandomErasingExperiment("exp_no_random_erasing", args)
    experiment.run()