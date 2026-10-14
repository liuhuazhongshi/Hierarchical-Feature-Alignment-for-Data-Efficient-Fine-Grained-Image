#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
消融实验：原始模型（Baseline）
======================================================================

包含所有功能的原始模型作为基准：
- 双分支结构（超类分支 + 细粒度分支）
- 层次一致性损失（KL）
- 正交损失
- CutMix数据增强
- SAM优化器

Author : Like Li
Date   : 2025-12-05
======================================================================
"""
import sys
import os
sys.path.insert(0, os.path.abspath("./"))
from model.experiment import BaseExperiment
from model.args import parse_args

if __name__ == "__main__":
    args = parse_args()
    experiment = BaseExperiment("exp_baseline", args)
    experiment.run()
