#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力机制消融实验 - 配置2：SE（通道注意力）
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from base_experiment import BaseExperiment, parse_args


class AttentionExperiment2(BaseExperiment):
    """注意力机制消融实验 - 配置2"""
    
    def __init__(self):
        # 解析命令行参数
        args = parse_args()
        
        # 设置消融实验的默认参数
        args.use_coarse_branch = True
        args.use_cross_attention = True  # 需要启用，以便在内部选择SE注意力
        args.attention_type = "se"  # SE通道注意力
        args.use_kl = True
        args.use_dynamic_kl = True
        args.use_sam = True
        
        # 调用父类初始化方法
        super().__init__(experiment_name="attention_2_se", args=args)


if __name__ == "__main__":
    # 创建实验实例并运行
    experiment = AttentionExperiment2()
    experiment.run()