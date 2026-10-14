#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据规模消融实验模板
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from base_experiment import BaseExperiment, parse_args


class DataAblationExperiment(BaseExperiment):
    """数据规模消融实验基础类"""
    
    def __init__(self, train_ratio, use_sam, exp_name):
        # 解析命令行参数
        args = parse_args()
        
        # 设置实验参数
        args.use_coarse_branch = True  # 使用粗粒度分支
        args.use_cross_attention = True  # 使用跨分支注意力
        args.use_kl = False  # 不使用KL散度
        args.use_dynamic_kl = False
        args.use_sam = use_sam
        args.train_ratio = train_ratio  # 设置训练集比例
        args.use_cutmix = True  # 使用CutMix增强
        
        # 设置适合的数据增强策略
        args.strong_aug = True
        
        # 设置学习率
        if use_sam:
            args.lr = 0.1  # SAM需要不同的学习率
        else:
            args.lr = 0.001  # 普通AdamW的学习率
        
        # 调用父类初始化方法
        super().__init__(experiment_name=exp_name, args=args)


# 训练集比例和是否使用SAM的组合
train_ratios = [0.3, 0.5, 0.8]
use_sam_options = [True, False]

# 创建所有实验配置
if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据规模消融实验')
    parser.add_argument('--ratio', type=float, required=True, choices=[0.3, 0.5, 0.8], help='训练集比例')
    parser.add_argument('--sam', type=int, required=True, choices=[0, 1], help='是否使用SAM (0: 不使用, 1: 使用)')
    args = parser.parse_args()
    
    # 创建实验名称
    exp_name = f"data_ablation_{int(args.ratio*100)}_sam{args.sam}"
    
    # 创建实验实例并运行
    experiment = DataAblationExperiment(args.ratio, bool(args.sam), exp_name)
    experiment.run()
