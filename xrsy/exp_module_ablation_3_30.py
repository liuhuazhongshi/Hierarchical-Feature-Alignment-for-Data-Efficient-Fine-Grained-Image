#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置3：有粗粒度分支、有跨分支注意力、无动态KL、无SAM - 30%训练集
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from base_experiment import BaseExperiment, parse_args


class ModuleAblationExperiment3_30(BaseExperiment):
    """模块级消融实验 - 配置3 - 30%训练集"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数
        args.use_coarse_branch = True
        args.use_cross_attention = True
        args.use_kl = False  # 注意：这里先不使用动态KL
        args.use_dynamic_kl = False
        args.use_sam = False
        args.train_ratio = 0.3  # 30%训练集
        args.warm = 10  # 设置线性预热的epoch数为10
        args.drop = 0.1  # 设置dropout率为0.1
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_3_30", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment3_30()
    experiment.run()
