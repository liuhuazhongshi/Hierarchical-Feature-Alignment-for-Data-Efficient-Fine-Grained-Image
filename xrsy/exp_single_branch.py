#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
消融实验：单分支模型（Single Branch）
======================================================================

单分支模型，只保留细粒度分类分支：
- 单分支结构（仅细粒度分支）
- 无层次一致性损失（KL）
- 无正交损失
- 保留CutMix数据增强
- 保留SAM优化器

Author : Like Li
Date   : 2025-12-05
======================================================================
"""
from base_experiment import BaseExperiment, parse_args, AverageMeter, tqdm, F, torch, WideBasic
import torch.nn as nn

# 单分支WideResNet模型
class SingleBranchWideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, nc_fine=100, dropout=0.0):
        super().__init__()
        n = (depth - 4) // 6
        k = widen
        nStages = [16, 16*k, 32*k, 64*k]
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(WideBasic, nStages[1], n, 1, dropout)
        self.layer2 = self._make_layer(WideBasic, nStages[2], n, 2, dropout)
        self.layer3 = self._make_layer(WideBasic, nStages[3], n, 2, dropout)
        self.bn = nn.BatchNorm2d(nStages[3])

        # 仅保留细粒度分支
        self.fine_dec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fine_cls = nn.Linear(nStages[3], nc_fine)

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        layers = [block(self.in_planes, planes, stride, dropout)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))

        feat = out
        ff = self.fine_dec(feat)
        logit_f = self.fine_cls(ff)
        # 返回与双分支模型兼容的格式
        return None, logit_f, None, ff

    def hier_kl(self, logit_f, logit_c, lam):
        """兼容接口，返回0"""
        return 0.0

    def orth_loss(self, fs, ff, lam):
        """兼容接口，返回0"""
        return 0.0

class SingleBranchExperiment(BaseExperiment):
    def run(self):
        """运行单分支实验"""
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Arguments: {self.args}")
        
        # 设置随机种子
        self.set_seed(42)
        
        # 获取数据加载器
        train_loader, test_loader = self.get_data_loader(self.args.batch)
        
        # 创建单分支模型
        model = SingleBranchWideResNet(depth=28, widen=10, dropout=self.args.drop).cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # 获取优化器和调度器
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)
        
        best_acc = 0.0
        start_epoch = 1
        
        # 训练循环
        for epoch in range(start_epoch, self.args.epoch + 1):
            if epoch <= self.args.warm:   # linear warm-up
                for g in optimizer.param_groups:
                    g["lr"] = self.args.lr * epoch / self.args.warm

            # 训练一个epoch
            train_loss, train_acc_s, train_acc_f = self.train_one_epoch(model, train_loader, optimizer, epoch)
            
            # 更新学习率
            if epoch > self.args.warm:
                scheduler.step()
            
            # 测试模型
            test_acc_s, test_acc_f = self.test(model, test_loader, epoch)
            
            # 保存最佳模型
            is_best = test_acc_f > best_acc
            best_acc = max(best_acc, test_acc_f)
            
            # 写入CSV
            self.write_to_csv(epoch, train_loss, train_acc_s, train_acc_f, test_acc_s, test_acc_f, best_acc)
            
            # 保存检查点
            state = {
                "model": model.state_dict(), 
                "optimizer": optimizer.state_dict(), 
                "scheduler": scheduler.state_dict(), 
                "epoch": epoch, 
                "best_acc": best_acc
            }
            torch.save(state, f"{self.args.save_dir}/{self.experiment_name}_latest.pth")
            if is_best:
                torch.save(state, f"{self.args.save_dir}/{self.experiment_name}_best.pth")
                self.logger.info(f"*** Best model updated: {best_acc:.4f} ***")
        
        self.logger.info("All done!")
        self.logger.info(f"Best accuracy: {best_acc:.4f}")
    
    def train_one_epoch(self, model, loader, optimizer, epoch):
        """训练一个epoch，仅使用细粒度分支"""
        model.train()
        losses, acc_s, acc_f = AverageMeter(), AverageMeter(), AverageMeter()
        pbar = tqdm(loader, total=len(loader), leave=False, desc=f"E{epoch}")

        for x, y_a, y_b, lam in pbar:
            x, y_a, y_b = x.cuda(), y_a.cuda(), y_b.cuda()

            # 1. 正常前向 + 反向，生成梯度
            def closure():
                _, logit_f, _, _ = model(x)
                loss_ce_f = lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)
                return loss_ce_f

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
                _, logit_f, _, _ = model(x)
                pred_f = logit_f.argmax(1)
                acc_f_batch = lam * (pred_f == y_a).float().mean() + (1 - lam) * (pred_f == y_b).float().mean()
                acc_f.update(acc_f_batch.item(), x.size(0))
                losses.update(loss.item(), x.size(0))
            pbar.set_postfix(loss=losses.avg, acc_f=acc_f.avg)

        self.logger.info(f"Train E{epoch}: loss={losses.avg:.4f}  acc_f={acc_f.avg:.4f}")
        return losses.avg, 0.0, acc_f.avg
    
    @torch.no_grad()
    def test(self, model, loader, epoch):
        """测试单分支模型"""
        model.eval()
        correct_f = total = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            _, logit_f, _, _ = model(x)
            correct_f += (logit_f.argmax(1) == y).sum().item()
            total += x.size(0)
        acc_f = correct_f / total
        self.logger.info(f"Test  E{epoch}: acc_f={acc_f:.4f}")
        return 0.0, acc_f

if __name__ == "__main__":
    args = parse_args()
    experiment = SingleBranchExperiment("exp_single_branch", args)
    experiment.run()
