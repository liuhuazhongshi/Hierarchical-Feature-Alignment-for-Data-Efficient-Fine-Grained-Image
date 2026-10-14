#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化器模块
"""
import torch
import torch.optim as optim

class SAM(optim.Optimizer):
    """Sharpness-Aware Minimization (SAM)优化器"""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """SAM优化器第一步更新"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """SAM优化器第二步更新"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        """计算梯度范数"""
        return torch.norm(torch.stack([p.grad.norm(p=2) for group in self.param_groups
                                      for p in group["params"] if p.grad is not None]), p=2)
