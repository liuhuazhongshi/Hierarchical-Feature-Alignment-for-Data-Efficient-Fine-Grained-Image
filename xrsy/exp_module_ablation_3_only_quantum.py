#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置3：有粗粒度分支、有跨分支注意力、无动态KL、无SAM、只启用量子纠缠模块
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.experiment import BaseExperiment
from model.args import parse_args


class ModuleAblationExperiment3OnlyQuantum(BaseExperiment):
    """模块级消融实验 - 配置3：只启用量子纠缠模块"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数
        args.use_coarse_branch = True
        args.use_cross_attention = True
        args.use_kl = False  # 注意：这里先不使用动态KL
        args.use_dynamic_kl = False
        args.use_sam = False
        
        # 只启用量子纠缠模块
        args.use_innovation = True
        args.use_quantum_entanglement = True
        args.use_causal_decoupling = False
        args.use_memory_network = False
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_3_only_quantum", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment3OnlyQuantum()
    experiment.run()
