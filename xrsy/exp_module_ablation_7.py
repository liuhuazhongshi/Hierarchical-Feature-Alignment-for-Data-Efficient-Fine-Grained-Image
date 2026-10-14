#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置7：有粗粒度分支、无跨分支注意力、无KL、有SAM
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from base_experiment import BaseExperiment, parse_args


class ModuleAblationExperiment7(BaseExperiment):
    """模块级消融实验 - 配置7"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数 - 开启粗粒度分支和跨分支注意力
        args.use_coarse_branch = True
        args.use_cross_attention = True
        args.use_kl = False
        args.use_dynamic_kl = False
        args.use_sam = True
        
        # 注意力机制相关参数
        args.attention_type = "cross_branch"  # 确保使用最新的跨分支注意力机制
        
        # SAM需要不同的学习率
        args.lr = 0.1  # 恢复SAM所需的初始学习率
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_7", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment7()
    experiment.run()
