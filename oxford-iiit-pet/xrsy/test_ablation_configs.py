#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证消融实验配置正确性
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.args import parse_args

# 导入所有消融实验类
from exp_module_ablation_3_no_innovation import ModuleAblationExperiment3NoInnovation
from exp_module_ablation_3_only_quantum import ModuleAblationExperiment3OnlyQuantum
from exp_module_ablation_3_only_causal import ModuleAblationExperiment3OnlyCausal
from exp_module_ablation_3_only_memory import ModuleAblationExperiment3OnlyMemory
from exp_module_ablation_3_full_innovation import ModuleAblationExperiment3FullInnovation

def test_config(experiment_class, expected_config):
    """测试实验配置是否符合预期"""
    print(f"\nTesting {experiment_class.__name__}...")
    
    # 创建实验实例（不运行训练）
    experiment = experiment_class()
    
    # 获取配置
    args = experiment.args
    
    # 验证配置
    all_correct = True
    for key, expected_value in expected_config.items():
        actual_value = getattr(args, key)
        if actual_value != expected_value:
            print(f"  ERROR: {key} = {actual_value}, expected {expected_value}")
            all_correct = False
        else:
            print(f"  OK: {key} = {actual_value}")
    
    return all_correct

def main():
    """主测试函数"""
    print("验证消融实验配置正确性...")
    
    # 定义每个实验的预期配置
    test_cases = [
        # 无创新模块
        {
            'experiment_class': ModuleAblationExperiment3NoInnovation,
            'expected_config': {
                'use_coarse_branch': True,
                'use_cross_attention': True,
                'use_kl': False,
                'use_dynamic_kl': False,
                'use_sam': False,
                'use_innovation': False,
                'use_quantum_entanglement': False,
                'use_causal_decoupling': False,
                'use_memory_network': False
            }
        },
        # 只启用量子纠缠模块
        {
            'experiment_class': ModuleAblationExperiment3OnlyQuantum,
            'expected_config': {
                'use_coarse_branch': True,
                'use_cross_attention': True,
                'use_kl': False,
                'use_dynamic_kl': False,
                'use_sam': False,
                'use_innovation': True,
                'use_quantum_entanglement': True,
                'use_causal_decoupling': False,
                'use_memory_network': False
            }
        },
        # 只启用因果解耦模块
        {
            'experiment_class': ModuleAblationExperiment3OnlyCausal,
            'expected_config': {
                'use_coarse_branch': True,
                'use_cross_attention': True,
                'use_kl': False,
                'use_dynamic_kl': False,
                'use_sam': False,
                'use_innovation': True,
                'use_quantum_entanglement': False,
                'use_causal_decoupling': True,
                'use_memory_network': False
            }
        },
        # 只启用记忆网络模块
        {
            'experiment_class': ModuleAblationExperiment3OnlyMemory,
            'expected_config': {
                'use_coarse_branch': True,
                'use_cross_attention': True,
                'use_kl': False,
                'use_dynamic_kl': False,
                'use_sam': False,
                'use_innovation': True,
                'use_quantum_entanglement': False,
                'use_causal_decoupling': False,
                'use_memory_network': True
            }
        },
        # 全启用创新模块
        {
            'experiment_class': ModuleAblationExperiment3FullInnovation,
            'expected_config': {
                'use_coarse_branch': True,
                'use_cross_attention': True,
                'use_kl': False,
                'use_dynamic_kl': False,
                'use_sam': False,
                'use_innovation': True,
                'use_quantum_entanglement': True,
                'use_causal_decoupling': True,
                'use_memory_network': True
            }
        }
    ]
    
    # 运行所有测试
    all_passed = True
    for test_case in test_cases:
        if not test_config(test_case['experiment_class'], test_case['expected_config']):
            all_passed = False
    
    # 打印最终结果
    print("\n" + "="*50)
    if all_passed:
        print("✅ 所有消融实验配置验证通过！")
    else:
        print("❌ 部分消融实验配置验证失败！")
    print("="*50)

if __name__ == "__main__":
    main()
