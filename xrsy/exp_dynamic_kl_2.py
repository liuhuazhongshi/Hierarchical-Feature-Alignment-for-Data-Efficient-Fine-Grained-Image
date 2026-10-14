#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic KL专项消融实验 - 配置2：静态KL (λ=1.0)
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from base_experiment import BaseExperiment, parse_args


class DynamicKLExperiment2(BaseExperiment):
    """Dynamic KL专项消融实验 - 配置2"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数
        args.use_coarse_branch = True
        args.use_cross_attention = True
        args.use_kl = True  # 使用KL
        args.use_dynamic_kl = False  # 静态KL
        args.lambda_kl = 1.0  # 固定λ=1.0
        args.use_sam = True
        
        # 调用父类初始化方法
        super().__init__(experiment_name="dynamic_kl_2_static_kl", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = DynamicKLExperiment2()
    experiment.run()