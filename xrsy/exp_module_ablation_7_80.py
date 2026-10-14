#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置7：有粗粒度分支、无跨分支注意力、无KL、有SAM - 80%训练集
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from base_experiment import BaseExperiment, parse_args


class ModuleAblationExperiment7_80(BaseExperiment):
    """模块级消融实验 - 配置7 - 80%训练集"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数
        args.use_coarse_branch = True
        args.use_cross_attention = False
        args.use_kl = False
        args.use_dynamic_kl = False
        args.use_sam = True
        
        # 设置训练集比例
        args.train_ratio = 0.8
        
        # SAM需要不同的学习率
        args.lr = 0.1  # 恢复SAM所需的初始学习率
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_7_80", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment7_80()
    experiment.run()