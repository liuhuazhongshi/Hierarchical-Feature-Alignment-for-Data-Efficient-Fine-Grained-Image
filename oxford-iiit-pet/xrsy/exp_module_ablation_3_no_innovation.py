#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置3：有粗粒度分支、有跨分支注意力、无动态KL、无SAM、无创新模块
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import os
sys.path.insert(0, os.path.abspath("./"))
from model.experiment import BaseExperiment
from model.args import parse_args


class ModuleAblationExperiment3NoInnovation(BaseExperiment):
    """模块级消融实验 - 配置3：无创新模块"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数
        args.use_coarse_branch = True
        args.use_cross_attention = True
        args.use_kl = False  # 注意：这里先不使用动态KL
        args.use_dynamic_kl = False
        args.use_sam = False
        
        # 禁用所有创新模块
        args.use_innovation = False
        args.use_quantum_entanglement = False
        args.use_causal_decoupling = False
        args.use_memory_network = False
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_3_no_innovation", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment3NoInnovation()
    experiment.run()
