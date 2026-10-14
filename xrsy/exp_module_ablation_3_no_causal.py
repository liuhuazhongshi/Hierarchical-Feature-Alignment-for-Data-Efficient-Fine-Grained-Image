#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置3-2：基础配置 + 禁用因果特征解耦模块
基础配置：有粗粒度分支、有跨分支注意力、无动态KL、无SAM
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.experiment import BaseExperiment
from model.args import parse_args


class ModuleAblationExperiment3NoCausal(BaseExperiment):
    """模块级消融实验 - 配置3-2：禁用因果特征解耦模块"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数（与配置3相同）
        args.use_coarse_branch = True
        args.use_cross_attention = True
        args.use_kl = False
        args.use_dynamic_kl = False
        args.use_sam = False
        
        # 禁用因果特征解耦模块
        args.use_causal_decoupling = False
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_3_no_causal", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment3NoCausal()
    experiment.run()
