#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置5：大部分参数为False，仅SAM为True - 80%训练集
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


class ModuleAblationExperiment5_80(BaseExperiment):
    """模块级消融实验 - 配置5 - 80%训练集"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数 - 大部分为False，仅SAM为True
        args.use_coarse_branch = False
        args.use_cross_attention = False
        args.use_kl = False
        args.use_dynamic_kl = False
        args.use_sam = True
        
        # 注意力机制相关参数 - 使用最新的跨分支注意力实现
        args.attention_type = "cross_branch"
        
        # 设置训练集比例
        args.train_ratio = 0.8
        
        # SAM需要不同的学习率
        args.lr = 0.1  # 恢复SAM所需的初始学习率
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_5_80", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment5_80()
    experiment.run()