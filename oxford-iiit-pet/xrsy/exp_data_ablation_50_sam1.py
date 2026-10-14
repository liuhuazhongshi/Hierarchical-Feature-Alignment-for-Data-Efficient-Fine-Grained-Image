#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据规模消融实验 - 配置：50%训练集、有SAM
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import sys
import os
sys.path.insert(0, os.path.abspath("./"))
from model.experiment import BaseExperiment
from model.args import parse_args


class DataAblationExperiment50SAM1(BaseExperiment):
    """数据规模消融实验 - 50%训练集、有SAM"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置实验参数
        args.use_coarse_branch = True  # 使用粗粒度分支
        args.use_cross_attention = True  # 使用跨分支注意力
        args.use_kl = False  # 不使用KL散度
        args.use_dynamic_kl = False
        args.use_sam = True  # 使用SAM
        args.train_ratio = 0.5  # 设置训练集比例为50%
        args.use_cutmix = True  # 使用CutMix增强
        
        # 设置适合的数据增强策略
        args.strong_aug = True
        
        # SAM需要不同的学习率
        args.lr = 0.1
        
        # 调用父类初始化方法
        super().__init__(experiment_name="data_ablation_50_sam1", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = DataAblationExperiment50SAM1()
    experiment.run()