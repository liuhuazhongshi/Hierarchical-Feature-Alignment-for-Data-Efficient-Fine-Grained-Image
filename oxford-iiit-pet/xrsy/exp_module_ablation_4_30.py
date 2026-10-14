#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置5: 有粗粒度分支、有跨分支注意力、改进的动态KL、无SAM - 30%训练集
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


class ModuleAblationExperiment5_30(BaseExperiment):
    """模块级消融实验 - 配置5 - 30%训练集"""

    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置改进的实验参数
        args.use_coarse_branch = True
        args.use_cross_attention = True
        args.use_kl = True
        args.use_dynamic_kl = True
        args.use_sam = False
        args.train_ratio = 0.3  # 30%训练集
        args.warm = 10  # 设置线性预热的epoch数为10
        args.drop = 0.1  # 设置dropout率为0.1
        
        # 改进的动态KL参数
        args.lambda_kl = 0.5  # 增加KL权重，从0.2提升到0.5
        args.kl_warm_epochs = 10  # 减少预热期，从20降到10
        args.kl_strategy = "logistic"  # 使用新的逻辑增长策略
        args.kl_power = 2.0  # 增加指数增长的幂次
        
        # 可选：调整或去除正交损失
        args.lambda_orth = 1e-4  # 减少正交损失权重，从5e-4降到1e-4
        # args.no_reg = True  # 如果需要可以完全关闭正则化
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_5_30", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment5_30()
    experiment.run()
