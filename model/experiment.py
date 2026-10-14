#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验类模块
"""
import os
import math
import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .logger import get_logger
from .utils import set_seed
from .data import CIFAR100_MEAN, CIFAR100_STD, cutmix_collate
from .model import WideResNet
from .optimizer import SAM
from .metrics import AverageMeter
from torchvision.transforms import RandomErasing, AutoAugment, AutoAugmentPolicy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class BaseExperiment:
    """基础实验类，其他消融实验将继承此类"""
    def __init__(self, experiment_name, args):
        """初始化实验"""
        self.experiment_name = experiment_name
        self.args = args
        self.logger = get_logger(experiment_name, f"logs/{experiment_name}.log")
        self.csv_file = f"csv/{experiment_name}.csv"
        
        # 设置随机种子，确保实验可复现
        seed = getattr(args, 'seed', 42)
        set_seed(seed)
        self.logger.info(f"设置随机种子: {seed}")
        
        # 创建保存目录
        Path(args.save_dir).mkdir(exist_ok=True)
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=args.save_dir)
        self.logger.info(f"Initializing TensorBoard with log dir: {args.save_dir}")
        
        # 初始化CSV文件
        self._init_csv()
        
    def _init_csv(self):
        """初始化CSV文件，写入表头"""
        # 创建CSV目录
        csv_dir = os.path.dirname(self.csv_file)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
            self.logger.info(f"创建CSV目录: {csv_dir}")
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc_s', 'train_acc_f', 'test_acc_s', 'test_acc_f', 'best_acc'])
    
    def write_to_csv(self, epoch, train_loss, train_acc_s, train_acc_f, test_acc_s, test_acc_f, best_acc):
        """将实验结果写入CSV文件"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc_s, train_acc_f, test_acc_s, test_acc_f, best_acc])
    
    def get_data_loader(self, batch_size, use_cutmix=False):
        """获取数据加载器"""
        # 根据strong_aug参数选择不同的数据增强策略
        if self.args.strong_aug:
            # 更强的数据增强
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),  # 使用AutoAugment代替RandAugment
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
                RandomErasing(p=0.5, scale=(0.02, 0.15))
            ])
        else:
            # 默认数据增强
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),  # 使用AutoAugment
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
                RandomErasing(p=0.5, scale=(0.02, 0.15))
            ])
        
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
        
        train_set = datasets.CIFAR100(root="./data", train=True, download=False, transform=train_tf)
        test_set  = datasets.CIFAR100(root="./data", train=False, download=False, transform=test_tf)
        
        # 应用训练集比例采样
        if hasattr(self.args, 'train_ratio') and self.args.train_ratio < 1.0:
            # 计算采样数量
            num_samples = int(len(train_set) * self.args.train_ratio)
            # 创建随机索引，使用与实验一致的种子
            seed = getattr(self.args, 'seed', 42)
            import numpy as np
            np.random.seed(seed)
            indices = np.random.choice(len(train_set), num_samples, replace=False)
            # 使用Subset创建采样后的数据集
            train_set = torch.utils.data.Subset(train_set, indices)
            self.logger.info(f"使用训练集比例: {self.args.train_ratio:.1%}, 采样数量: {len(train_set)}, 采样种子: {seed}")
        
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=self.args.num_workers, 
            pin_memory=True, 
            drop_last=True,
            collate_fn=cutmix_collate if use_cutmix else None
        )
        
        test_loader = DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.args.num_workers, 
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def get_optimizer(self, model):
        """获取优化器"""
        if self.args.use_sam:
            base_opt = torch.optim.SGD
            optimizer = SAM(
                model.parameters(), 
                base_opt, 
                lr=self.args.lr, 
                momentum=0.9, 
                weight_decay=self.args.wd
            )
        else:
            # 使用AdamW优化器
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.args.lr, 
                weight_decay=self.args.wd
            )
        return optimizer

    def get_scheduler(self, optimizer):
        """获取学习率调度器"""
        def sche(e): return 0.5 * (1 + math.cos(math.pi * e / self.args.epoch))
        # 适配不同的优化器
        if self.args.use_sam:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, sche)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=200
            )
        return scheduler

    def train_one_epoch(self, model, loader, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        losses, acc_s, acc_f = AverageMeter(), AverageMeter(), AverageMeter()
        pbar = tqdm(loader, total=len(loader), leave=False, desc=f"E{epoch}")

        for batch in pbar:
            # 兼容带CutMix和不带CutMix的数据格式
            if len(batch) == 2:
                # 普通数据格式: (x, y)
                x, y = batch
                lam = 1.0
                y_a, y_b = y, y
            else:
                # CutMix数据格式: (x, y_a, y_b, lam)
                x, y_a, y_b, lam = batch
                
            x, y_a, y_b = x.cuda(), y_a.cuda(), y_b.cuda()
            y_c_a, y_c_b = model.f2c[y_a], model.f2c[y_b]

            # 动态KL权重（改进的自适应调整策略）
            if self.args.use_dynamic_kl:
                if epoch <= self.args.kl_warm_epochs:
                    # 改进的warm-up策略：使用更激进的增长方式
                    if self.args.kl_strategy == "linear":
                        # 基础线性增长
                        lambda_kl = self.args.lambda_kl * epoch / self.args.kl_warm_epochs
                    elif self.args.kl_strategy == "cosine":
                        # 改进的余弦增长：更激进的增长
                        lambda_kl = self.args.lambda_kl * (1 - math.cos(math.pi * epoch / self.args.kl_warm_epochs))
                    elif self.args.kl_strategy == "exponential":
                        # 改进的指数增长：更快达到最大权重
                        lambda_kl = self.args.lambda_kl * (epoch / self.args.kl_warm_epochs) ** (self.args.kl_power * 2)
                    elif self.args.kl_strategy == "logistic":
                        # 新增逻辑增长策略：S型曲线，更平滑的过渡
                        t = epoch / self.args.kl_warm_epochs
                        lambda_kl = self.args.lambda_kl / (1 + math.exp(-10 * (t - 0.5)))
                else:
                    # 训练后期：保持稳定权重
                    lambda_kl = self.args.lambda_kl
            else:
                lambda_kl = self.args.lambda_kl

            # 1. 前向传播
            if model.use_coarse_branch:
                logit_s, logit_f, fs, ff = model(x)
                loss_ce_s = lam * F.cross_entropy(logit_s, y_c_a) + (1 - lam) * F.cross_entropy(logit_s, y_c_b)
                loss_ce_f = lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)
                
                # 计算正则化损失
                if not self.args.no_reg and model.use_kl:
                    loss_kl = model.hier_kl(logit_f, logit_s, lambda_kl)
                    loss_orth = model.orth_loss(fs, ff, self.args.lambda_orth)
                    loss = loss_ce_s + loss_ce_f + loss_kl + loss_orth
                else:
                    loss = loss_ce_s + loss_ce_f
            else:
                # 只有细粒度分支
                logit_f = model(x)
                loss_ce_f = lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)
                loss = loss_ce_f

            # 2. 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            
            if self.args.use_sam:
                # SAM优化器的两步更新
                optimizer.first_step(zero_grad=True)
                
                # 第二次前向传播和反向传播
                if model.use_coarse_branch:
                    logit_s, logit_f, fs, ff = model(x)
                    loss_ce_s = lam * F.cross_entropy(logit_s, y_c_a) + (1 - lam) * F.cross_entropy(logit_s, y_c_b)
                    loss_ce_f = lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)
                    
                    if not self.args.no_reg and model.use_kl:
                        loss_kl = model.hier_kl(logit_f, logit_s, lambda_kl)
                        loss_orth = model.orth_loss(fs, ff, self.args.lambda_orth)
                        loss = loss_ce_s + loss_ce_f + loss_kl + loss_orth
                    else:
                        loss = loss_ce_s + loss_ce_f
                else:
                    logit_f = model(x)
                    loss_ce_f = lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)
                    loss = loss_ce_f
                
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                # 标准优化器更新
                optimizer.step()

            # 3. 记录指标
            with torch.no_grad():
                if model.use_coarse_branch:
                    logit_s, logit_f, _, _ = model(x)
                    pred_s = logit_s.argmax(1)
                    pred_f = logit_f.argmax(1)
                    acc_s_batch = lam * (pred_s == y_c_a).float().mean() + (1 - lam) * (pred_s == y_c_b).float().mean()
                    acc_f_batch = lam * (pred_f == y_a).float().mean() + (1 - lam) * (pred_f == y_b).float().mean()
                    acc_s.update(acc_s_batch.item(), x.size(0))
                    acc_f.update(acc_f_batch.item(), x.size(0))
                else:
                    logit_f = model(x)
                    pred_f = logit_f.argmax(1)
                    acc_f_batch = lam * (pred_f == y_a).float().mean() + (1 - lam) * (pred_f == y_b).float().mean()
                    acc_f.update(acc_f_batch.item(), x.size(0))
                
                losses.update(loss.item(), x.size(0))
                pbar.set_postfix(loss=losses.avg, acc_s=acc_s.avg if model.use_coarse_branch else 'N/A', acc_f=acc_f.avg)

        if model.use_coarse_branch:
            self.logger.info(f"Train E{epoch}: loss={losses.avg:.4f}  acc_s={acc_s.avg:.4f}  acc_f={acc_f.avg:.4f}")
        else:
            self.logger.info(f"Train E{epoch}: loss={losses.avg:.4f}  acc_f={acc_f.avg:.4f}")
        
        # 记录到TensorBoard
        self.writer.add_scalar('Train/Loss', losses.avg, epoch)
        if model.use_coarse_branch:
            self.writer.add_scalar('Train/Accuracy_Super', acc_s.avg, epoch)
        self.writer.add_scalar('Train/Accuracy_Fine', acc_f.avg, epoch)
        
        return losses.avg, acc_s.avg if model.use_coarse_branch else 0.0, acc_f.avg

    @torch.no_grad()
    def test(self, model, loader, epoch):
        """测试模型"""
        model.eval()
        correct_s = correct_f = total = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            y_c = model.f2c[y]
            
            if model.use_coarse_branch:
                logit_s, logit_f, _, _ = model(x)
                correct_s += (logit_s.argmax(1) == y_c).sum().item()
            else:
                logit_f = model(x)
            
            correct_f += (logit_f.argmax(1) == y).sum().item()
            total += x.size(0)
        
        if model.use_coarse_branch:
            acc_s, acc_f = correct_s / total, correct_f / total
            self.logger.info(f"Test  E{epoch}: acc_s={acc_s:.4f}  acc_f={acc_f:.4f}")
        else:
            acc_s, acc_f = 0.0, correct_f / total
            self.logger.info(f"Test  E{epoch}: acc_f={acc_f:.4f}")
        
        # 记录到TensorBoard
        if model.use_coarse_branch:
            self.writer.add_scalar('Test/Accuracy_Super', acc_s, epoch)
        self.writer.add_scalar('Test/Accuracy_Fine', acc_f, epoch)
        
        return acc_s, acc_f

    def run(self):
        """运行实验"""
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Arguments: {self.args}")
        
        # 设置随机种子
        set_seed(42)
        
        # 获取数据加载器
        train_loader, test_loader = self.get_data_loader(self.args.batch, use_cutmix=self.args.use_cutmix)
        
        # 创建模型
        model = WideResNet(
            depth=self.args.depth, 
            widen=self.args.widen, 
            dropout=self.args.drop,
            use_coarse_branch=self.args.use_coarse_branch,
            use_cross_attention=self.args.use_cross_attention,
            use_kl=self.args.use_kl,
            attention_type=getattr(self.args, 'attention_type', 'cross_branch')
        ).cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        # 获取优化器和调度器
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)
        
        best_acc = 0.0
        start_epoch = 1
        
        # 不加载任何检查点，始终从头开始训练
        
        # 训练循环
        for epoch in range(start_epoch, self.args.epoch + 1):
            if epoch <= self.args.warm:   # linear warm-up
                if self.args.use_sam:
                    for g in optimizer.param_groups:
                        g["lr"] = self.args.lr * epoch / self.args.warm
                else:
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
        
        # 关闭TensorBoard writer
        self.writer.close()
