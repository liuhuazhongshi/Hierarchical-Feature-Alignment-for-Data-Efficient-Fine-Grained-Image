#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
消融实验：无SAM优化器（No SAM Optimizer）
======================================================================

移除SAM优化器的模型，使用普通SGD：
- 双分支结构（超类分支 + 细粒度分支）
- 保留层次一致性损失（KL）
- 保留正交损失
- 保留CutMix数据增强
- 无SAM优化器（使用普通SGD）

Author : Like Li
Date   : 2025-12-05
======================================================================
"""
import torch.optim as optim
import sys
import os
sys.path.insert(0, os.path.abspath("./"))
from model.experiment import BaseExperiment
from model.args import parse_args

class NoSAMExperiment(BaseExperiment):
    def get_optimizer(self, model):
        """获取优化器，不使用SAM，使用普通SGD"""
        optimizer = optim.SGD(
            model.parameters(), 
            lr=self.args.lr, 
            momentum=0.9, 
            weight_decay=self.args.wd
        )
        return optimizer
    
    def train_one_epoch(self, model, loader, optimizer, epoch):
        """训练一个epoch，不使用SAM"""
        model.train()
        losses, acc_s, acc_f = AverageMeter(), AverageMeter(), AverageMeter()
        pbar = tqdm(loader, total=len(loader), leave=False, desc=f"E{epoch}")

        for x, y_a, y_b, lam in pbar:
            x, y_a, y_b = x.cuda(), y_a.cuda(), y_b.cuda()
            y_c_a, y_c_b = model.f2c[y_a], model.f2c[y_b]

            # 1. 正常前向 + 反向，生成梯度
            def closure():
                logit_s, logit_f, fs, ff = model(x)
                loss_ce_s = lam * F.cross_entropy(logit_s, y_c_a) + (1 - lam) * F.cross_entropy(logit_s, y_c_b)
                loss_ce_f = lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)
                
                # 计算正则化损失
                loss_kl = model.hier_kl(logit_f, logit_s, self.args.lambda_kl)
                loss_orth = model.orth_loss(fs, ff, self.args.lambda_orth)
                
                return loss_ce_s + loss_ce_f + loss_kl + loss_orth

            loss = closure()
            optimizer.zero_grad()
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 记录指标
            with torch.no_grad():
                logit_s, logit_f, _, _ = model(x)
                pred_s = logit_s.argmax(1)
                pred_f = logit_f.argmax(1)
                acc_s_batch = lam * (pred_s == y_c_a).float().mean() + (1 - lam) * (pred_s == y_c_b).float().mean()
                acc_f_batch = lam * (pred_f == y_a).float().mean() + (1 - lam) * (pred_f == y_b).float().mean()
                acc_s.update(acc_s_batch.item(), x.size(0))
                acc_f.update(acc_f_batch.item(), x.size(0))
                losses.update(loss.item(), x.size(0))
            pbar.set_postfix(loss=losses.avg, acc_s=acc_s.avg, acc_f=acc_f.avg)

        self.logger.info(f"Train E{epoch}: loss={losses.avg:.4f}  acc_s={acc_s.avg:.4f}  acc_f={acc_f.avg:.4f}")
        return losses.avg, acc_s.avg, acc_f.avg

if __name__ == "__main__":
    args = parse_args()
    experiment = NoSAMExperiment("exp_no_sam", args)
    experiment.run()
