#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指标模块
"""
class AverageMeter:
    """平均指标计算器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.val = self.sum = self.count = 0
    
    def update(self, val, n=1):
        """更新指标"""
        self.val = val
        self.sum += val * n
        self.count += n
    
    @property
    def avg(self):
        """获取平均指标"""
        return self.sum / self.count
