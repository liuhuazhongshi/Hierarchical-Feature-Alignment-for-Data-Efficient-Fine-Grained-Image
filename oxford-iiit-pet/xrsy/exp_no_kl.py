#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
消融实验：无层次一致性损失（No KL Loss）
======================================================================

移除层次一致性损失（KL）的模型：
- 双分支结构（超类分支 + 细粒度分支）
- 无层次一致性损失（KL）
- 保留正交损失
- 保留CutMix数据增强
- 保留SAM优化器

Author : Like Li
Date   : 2025-12-05
======================================================================
"""
import sys
import os
sys.path.insert(0, os.path.abspath("./"))
from model.experiment import BaseExperiment
from model.args import parse_args

class NoKLExperiment(BaseExperiment):
    def train_one_epoch(self, model, loader, optimizer, epoch):
        """训练一个epoch，不使用KL损失"""
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
                
                # 计算正则化损失 - 不使用KL损失
                loss_orth = model.orth_loss(fs, ff, self.args.lambda_orth)
                
                return loss_ce_s + loss_ce_f + loss_orth

            loss = closure()
            optimizer.zero_grad()
            loss.backward()  # 先算一次梯度

            # 2. SAM 第一步：不要清梯度
            optimizer.first_step(zero_grad=False)

            # 3. 第二次前向 + 反向
            closure().backward()

            # 4. SAM 第二步：真正更新参数，并清零梯度
            optimizer.second_step(zero_grad=True)

            # 5. 记录指标
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
    experiment = NoKLExperiment("exp_no_kl", args)
    experiment.run()
