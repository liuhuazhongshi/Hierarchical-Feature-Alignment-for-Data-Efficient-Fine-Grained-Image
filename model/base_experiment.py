#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
Hierarchically-Disentangled CIFAR-100 消融实验基础框架
======================================================================

基础实验框架，其他消融实验将继承此类

Author : Like Li
Date   : 2025-12-05
======================================================================
"""

# 导入所有模块化组件
from model.logger import get_logger
from model.utils import set_seed
from model.data import CIFAR100_MEAN, CIFAR100_STD, fine2coarse_id, cutmix_collate
from model.model import WideResNet
from model.optimizer import SAM
from model.metrics import AverageMeter
from model.experiment import BaseExperiment
from model.args import parse_args

# 主函数入口
if __name__ == "__main__":
    args = parse_args()
    experiment = BaseExperiment("hierarchical_cifar100", args)
    experiment.run()