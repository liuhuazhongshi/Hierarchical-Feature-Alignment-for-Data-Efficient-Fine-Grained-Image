#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级消融实验 - 配置2：有粗粒度分支、无跨分支注意力、无动态KL、无SAM
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.experiment import BaseExperiment
from model.args import parse_args


class ModuleAblationExperiment2(BaseExperiment):
    """模块级消融实验 - 配置2"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数
        args.use_coarse_branch = True
        args.use_cross_attention = False
        args.use_kl = False
        args.use_dynamic_kl = False
        args.use_sam = False
        
        # 调用父类初始化方法
        super().__init__(experiment_name="module_ablation_2", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = ModuleAblationExperiment2()
    experiment.run()