#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
消融实验：无CutMix数据增强（No CutMix）
======================================================================

移除CutMix数据增强的模型：
- 双分支结构（超类分支 + 细粒度分支）
- 保留层次一致性损失（KL）
- 保留正交损失
- 无CutMix数据增强
- 保留SAM优化器

Author : Like Li
Date   : 2025-12-05
======================================================================
"""
from base_experiment import BaseExperiment, parse_args, AverageMeter, tqdm, F, torch

class NoCutMixExperiment(BaseExperiment):
    def get_data_loader(self, batch_size, use_cutmix=False):
        """获取数据加载器，不使用CutMix"""
        return super().get_data_loader(batch_size, use_cutmix=False)
    
    def train_one_epoch(self, model, loader, optimizer, epoch):
        """训练一个epoch，不使用CutMix"""
        model.train()
        losses, acc_s, acc_f = AverageMeter(), AverageMeter(), AverageMeter()
        pbar = tqdm(loader, total=len(loader), leave=False, desc=f"E{epoch}")

        for x, y in pbar:
            x, y = x.cuda(), y.cuda()
            y_c = model.f2c[y]

            # 1. 正常前向 + 反向，生成梯度
            def closure():
                logit_s, logit_f, fs, ff = model(x)
                loss_ce_s = F.cross_entropy(logit_s, y_c)
                loss_ce_f = F.cross_entropy(logit_f, y)
                
                # 计算正则化损失
                loss_kl = model.hier_kl(logit_f, logit_s, self.args.lambda_kl)
                loss_orth = model.orth_loss(fs, ff, self.args.lambda_orth)
                
                return loss_ce_s + loss_ce_f + loss_kl + loss_orth

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
                acc_s_batch = (pred_s == y_c).float().mean()
                acc_f_batch = (pred_f == y).float().mean()
                acc_s.update(acc_s_batch.item(), x.size(0))
                acc_f.update(acc_f_batch.item(), x.size(0))
                losses.update(loss.item(), x.size(0))
            pbar.set_postfix(loss=losses.avg, acc_s=acc_s.avg, acc_f=acc_f.avg)

        self.logger.info(f"Train E{epoch}: loss={losses.avg:.4f}  acc_s={acc_s.avg:.4f}  acc_f={acc_f.avg:.4f}")
        return losses.avg, acc_s.avg, acc_f.avg

if __name__ == "__main__":
    args = parse_args()
    experiment = NoCutMixExperiment("exp_no_cutmix", args)
    experiment.run()
