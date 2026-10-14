#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置8：有粗粒度分支、有跨分支注意力、无KL、有SAM - 80%训练集
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from base_experiment import BaseExperiment, parse_args


class ModuleAblationExperiment8_80(BaseExperiment):
    """模块级消融实验 - 配置8 - 80%训练集"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数 - 开启粗粒度分支、跨分支注意力和动态KL
        args.use_coarse_branch = True
        args.use_cross_attention = True
        args.use_kl = True
        args.use_dynamic_kl = True
        args.use_sam = True
        
        # 注意力机制相关参数
        args.attention_type = "cross_branch"  # 确保使用最新的跨分支注意力机制
        
        # KL相关参数 - 使用改进后的动态KL实现
        args.lambda_kl = 0.5  # 增加KL权重以提高效果
        args.kl_warm_epochs = 10  # 缩短预热期
        args.kl_strategy = "logistic"  # 使用改进的logistic策略
        args.kl_power = 1.0  # 指数策略的幂次
        
        # 设置训练集比例
        args.train_ratio = 0.8
        
        # SAM需要不同的学习率
        args.lr = 0.1  # 恢复SAM所需的初始学习率
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_8_80", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment8_80()
    experiment.run()