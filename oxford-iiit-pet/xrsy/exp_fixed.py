#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
Hierarchically-Disentangled CIFAR-100 Trainer (Fixed Version)
======================================================================

This script trains a **two-branch disentangled network** on CIFAR-100:
----------------------------------------------------------------------
1. Shared backbone  -> 通用视觉特征
2. Super-class branch (20 classes)  -> 超类解耦器 + 分类器
3. Fine-grained branch (100 classes)-> 细粒度解耦器 + 分类器

Features
--------
• PyTorch 2.x compatible
• tqdm progress bars for both epochs and batches
• color-logs to console & log-file (logging + RichHandler)
• automatic CIFAR-100 download / mean-std normalization
• cosine LR scheduler with warm-up
• hierarchical consistency loss (KL) between 100-cls → 20-cls
• orthogonal regularization between two branch features
• checkpointing & best-model saving
• single GPU (DataParallel ready, just wrap model)

Usage
-----
$ python exp_fixed.py --epoch 200 --batch 128 --lr 0.1 --wd 5e-4 --lambda_kl 0.5 --lambda_orth 1e-3

Author : Like Li
Date   : 2025-12-04
======================================================================
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandAugment, RandomErasing
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tqdm import tqdm
from rich.logging import RichHandler
import logging
import math
import random, numpy as np

# ----------- 日志 -----------
def get_logger(log_file="train.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),  # console
        ],
    )
    return logging.getLogger(__name__)

logger = get_logger()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 以下两行保证卷积确定性，稍慢 < 5 %
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- 参 ----------
def parse_args():
    parser = argparse.ArgumentParser("双分支WRN-16-8 + SAM  CIFAR-100")
    parser.add_argument("--epoch", type=int, default=400, help="Total number of epochs to train")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--drop", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lambda_kl", type=float, default=0.2, help="层次KL散度损失的权重")
    parser.add_argument("--lambda_orth", type=float, default=5e-4, help="正交损失的权重")
    parser.add_argument("--warm", type=int, default=25, help="线性预热的epoch数")
    parser.add_argument("--no_reg", action="store_true", help="不使用正则化损失")
    parser.add_argument("--save_dir", type=str, default="ckpt", help="保存检查点的目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点文件路径")
    parser.add_argument("--num_workers", type=int, default=6, help="数据加载器的工作进程数")
    parser.add_argument("--use_cutmix", type=lambda x: x.lower() == 'true', default=True, help="是否使用CutMix数据增强")
    parser.add_argument("--strong_aug", action="store_true", default=False, help="是否使用更强的数据增强")
    return parser.parse_args()

args = parse_args()
Path(args.save_dir).mkdir(exist_ok=True)

# ---------- 数据 ----------
OXFORD_IIIT_PET_MEAN = (0.485, 0.456, 0.406)
OXFORD_IIIT_PET_STD  = (0.229, 0.224, 0.225)
import torch

# Oxford-IIIT Pet类别数量
NUM_CLASSES = 37

# 创建细粒度到粗粒度的映射（猫和狗）
# 0: cat, 1: dog
fine2coarse_id = torch.zeros(NUM_CLASSES, dtype=torch.long)
# 前10个类别是猫，其余是狗
for i in range(10):
    fine2coarse_id[i] = 0  # cat
for i in range(10, NUM_CLASSES):
    fine2coarse_id[i] = 1  # dog

cutmix_beta = 1.0

def cutmix_collate(batch):
    """返回 CutMix 后的 tensor + 两个 target + lam"""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    # 采样beta分布的lam值
    lam = torch.distributions.Beta(cutmix_beta, cutmix_beta).sample().item()
    bsz = images.size(0)
    index = torch.randperm(bsz)
    W, H = images.size(-1), images.size(-2)
    
    # 根据lam值计算裁剪区域大小，确保至少为1x1
    cut_rat = math.sqrt(1. - lam)
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))
    
    # 随机选择裁剪区域的左上角坐标，确保裁剪区域有效
    bbx1 = torch.randint(0, W - cut_w + 1, (1,)).item()
    bby1 = torch.randint(0, H - cut_h + 1, (1,)).item()
    bbx2 = bbx1 + cut_w
    bby2 = bby1 + cut_h
    
    # 执行CutMix操作
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    
    # 计算实际的lam值
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)
    
    return images, labels, labels[index], lam

def get_loader(batch_size, num_workers=6, use_cutmix=True):
    if args.strong_aug:
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(OXFORD_IIIT_PET_MEAN, OXFORD_IIIT_PET_STD),
            RandomErasing(p=0.5, scale=(0.02, 0.15))
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(OXFORD_IIIT_PET_MEAN, OXFORD_IIIT_PET_STD),
            RandomErasing(p=0.5, scale=(0.02, 0.15))
        ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(OXFORD_IIIT_PET_MEAN, OXFORD_IIIT_PET_STD)
    ])
    
    try:
        import datasets as hf_datasets
        from datasets import load_dataset
        from PIL import Image
        
        class OxfordIIITPetDataset(torch.utils.data.Dataset):
            def __init__(self, split="train", transform=None):
                self.dataset = load_dataset(
                    "oxford-iiit-pet", 
                    split=split, 
                    data_dir="./data/oxford-iiit-pet",
                    download=False
                )
                self.transform = transform
                
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                item = self.dataset[idx]
                image = item["image"].convert("RGB")
                label = item["label"]
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        train_set = OxfordIIITPetDataset(split="train", transform=train_tf)
        test_set = OxfordIIITPetDataset(split="test", transform=test_tf)
    except ImportError:
        # 如果无法加载huggingface datasets，使用torchvision的数据集
        train_set = datasets.ImageFolder(root="./data/oxford-iiit-pet/train", transform=train_tf)
        test_set = datasets.ImageFolder(root="./data/oxford-iiit-pet/test", transform=test_tf)
    
    # 根据use_cutmix参数决定是否使用CutMix
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              collate_fn=cutmix_collate if use_cutmix else None)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# ---------- WRN-16-8 ----------
class WideBasic(nn.Module):
    def __init__(self, c_in, c_out, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride, 1, bias=False)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Conv2d(c_in, c_out, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.drop(out)
        out = self.conv2(out)
        return out + self.shortcut(x)

class WideResNet(nn.Module):
    def __init__(self, depth=16, widen=8, nc_coarse=2, nc_fine=37, dropout=0.0):
        super().__init__()
        # 添加模型深度合法性检查
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must be of the form 6n+4 (e.g., 28, 40)")
            
        n = (depth - 4) // 6
        k = widen
        nStages = [16, 16*k, 32*k, 64*k]
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(WideBasic, nStages[1], n, 1, dropout)
        self.layer2 = self._make_layer(WideBasic, nStages[2], n, 2, dropout)
        self.layer3 = self._make_layer(WideBasic, nStages[3], n, 2, dropout)
        self.bn = nn.BatchNorm2d(nStages[3])
        
        # 保存细粒度到粗粒度的映射关系
        self.f2c = fine2coarse_id.cuda()

        # 添加分支分离的特征提取层 - 设计不同结构以实现更好的特征解耦
        # 超类分支：使用更大的卷积核和更少的非线性变换，专注于全局特征
        self.super_branch = nn.Sequential(
            nn.Conv2d(nStages[3], nStages[3], 5, 1, 2, bias=False),  # 使用5x5卷积核
            nn.BatchNorm2d(nStages[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 细粒度分支：使用更小的卷积核和更多的非线性变换，专注于细节特征
        self.fine_branch = nn.Sequential(
            nn.Conv2d(nStages[3], nStages[3], 1, 1, 0, bias=False),
            nn.BatchNorm2d(nStages[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased dropout
            nn.Conv2d(nStages[3], nStages[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(nStages[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased dropout
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 定义多头注意力机制类
        class MultiHeadAttention(nn.Module):
            def __init__(self, embed_dim, num_heads=4, dropout=0.0):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                # 确保嵌入维度可以被头数整除
                assert embed_dim % num_heads == 0, "嵌入维度必须能被头数整除"
                
                self.head_dim = embed_dim // num_heads
                self.scale = self.head_dim ** -0.5
                
                # 线性变换层
                self.q_proj = nn.Linear(embed_dim, embed_dim)
                self.k_proj = nn.Linear(embed_dim, embed_dim)
                self.v_proj = nn.Linear(embed_dim, embed_dim)
                self.out_proj = nn.Linear(embed_dim, embed_dim)
                
                self.dropout = nn.Dropout(dropout)
                self.softmax = nn.Softmax(dim=-1)
            
            def forward(self, x):
                # x: [batch_size, embed_dim]
                batch_size = x.size(0)
                
                # 线性变换并分割为多头
                q = self.q_proj(x).view(batch_size, self.num_heads, self.head_dim)
                k = self.k_proj(x).view(batch_size, self.num_heads, self.head_dim)
                v = self.v_proj(x).view(batch_size, self.num_heads, self.head_dim)
                
                # 计算注意力分数
                # [batch_size, num_heads, head_dim] x [batch_size, num_heads, head_dim] -> [batch_size, num_heads]
                attn_scores = torch.sum(q * k, dim=-1) * self.scale
                attn_weights = self.softmax(attn_scores)
                attn_weights = self.dropout(attn_weights)
                
                # 应用注意力权重到值向量
                # [batch_size, num_heads] x [batch_size, num_heads, head_dim] -> [batch_size, num_heads, head_dim]
                attn_output = attn_weights.unsqueeze(-1) * v
                
                # 合并多头
                attn_output = attn_output.view(batch_size, self.embed_dim)
                
                # 最终线性变换
                output = self.out_proj(attn_output)
                
                return output
        
        # 添加分支间的交互机制 - 使用多头注意力机制
        self.attention_super = nn.Sequential(
            MultiHeadAttention(nStages[3], num_heads=4, dropout=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(nStages[3], nStages[3]),
            nn.Sigmoid()
        )
        
        self.attention_fine = nn.Sequential(
            MultiHeadAttention(nStages[3], num_heads=4, dropout=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(nStages[3], nStages[3]),
            nn.Sigmoid()
        )
        
        # 分类头
        self.super_cls = nn.Linear(nStages[3], nc_coarse)
        self.fine_cls = nn.Linear(nStages[3], nc_fine)

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        
        # 分支分离的特征提取
        fs_raw = self.super_branch(out)
        ff_raw = self.fine_branch(out)
        
        # 分支间的注意力交互
        # 超类分支关注细粒度分支的特征
        attention_s = self.attention_super(ff_raw)
        fs_attended = fs_raw * attention_s
        
        # 细粒度分支关注超类分支的特征
        attention_f = self.attention_fine(fs_raw)
        ff_attended = ff_raw * attention_f
        
        # 生成最终的分类logits
        logit_s = self.super_cls(fs_attended)
        logit_f = self.fine_cls(ff_attended)
        
        return logit_s, logit_f, fs_attended, ff_attended

    def hier_kl(self, logit_f, logit_c, lam):
        prob_f = F.softmax(logit_f, 1)
        # 使用更高效的索引操作替代for循环
        prob_c = torch.zeros_like(logit_c)
        prob_c = prob_c.scatter_add(1, self.f2c.expand(prob_f.size(0), -1).to(prob_f.device), prob_f)
        return lam * F.kl_div(F.log_softmax(logit_c, 1), prob_c, reduction='batchmean')

    def orth_loss(self, fs, ff, lam):
        """正交正则化：两个分支特征的L2内积的绝对值 + Gram矩阵正则化"""
        # 基础正交损失
        fs = F.normalize(fs, p=2, dim=1)
        ff = F.normalize(ff, p=2, dim=1)
        loss_inter = torch.abs((fs * ff).sum(1)).mean()
        
        # 增加Gram矩阵正则化，使同一分支内的特征也保持正交
        # 将特征reshape为batch_size × feature_dim
        batch_size = fs.size(0)
        fs_reshaped = fs.view(batch_size, -1)
        ff_reshaped = ff.view(batch_size, -1)
        
        # 计算Gram矩阵（batch_size × feature_dim × feature_dim）
        gram_s = torch.bmm(fs_reshaped.unsqueeze(2), fs_reshaped.unsqueeze(1))
        gram_f = torch.bmm(ff_reshaped.unsqueeze(2), ff_reshaped.unsqueeze(1))
        
        # 目标Gram矩阵：对角线为1，非对角线为0（完全正交）
        target = torch.eye(fs_reshaped.size(1), device=fs.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 计算Gram矩阵与目标矩阵的平方损失
        loss_gram_s = ((gram_s - target) ** 2).mean()
        loss_gram_f = ((gram_f - target) ** 2).mean()
        
        # 综合所有损失
        return lam * (loss_inter + loss_gram_s + loss_gram_f)

# ---------- SAM 优化器 ----------
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the traditional optimizers")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

# ---------- 训练和测试函数 ----------
def train_one_epoch(model, loader, optimizer, epoch, writer, use_cutmix=True):
    model.train()
    losses, acc_s, acc_f = AverageMeter(), AverageMeter(), AverageMeter()
    pbar = tqdm(loader, total=len(loader), leave=False, desc=f"E{epoch}")

    for batch in pbar:
        if use_cutmix:
            x, y_a, y_b, lam = batch
        else:
            x, y_a = batch
            y_b = y_a
            lam = 1.0
        
        x, y_a, y_b = x.cuda(), y_a.cuda(), y_b.cuda()
        y_c_a, y_c_b = model.f2c[y_a], model.f2c[y_b]

        # 动态KL权重（warm-up）
        if epoch <= args.warm:
            lambda_kl = args.lambda_kl * epoch / args.warm
        else:
            lambda_kl = args.lambda_kl

        # 1. 前向传播
        logit_s, logit_f, fs, ff = model(x)
        loss_ce_s = lam * F.cross_entropy(logit_s, y_c_a) + (1 - lam) * F.cross_entropy(logit_s, y_c_b)
        loss_ce_f = lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)
        loss = loss_ce_f + loss_ce_s

        # 2. 计算正则化损失
        if not args.no_reg:
            loss_kl = model.hier_kl(logit_f, logit_s, lambda_kl)
            loss_orth = model.orth_loss(fs, ff, args.lambda_orth)
            loss = loss + loss_kl + loss_orth
        else:
            loss_kl = 0
            loss_orth = 0

        # 3. 反向传播
        if args.use_sam is not None and args.use_sam:
            # SAM优化器的第一步
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # SAM优化器的第二步
            # 重新计算损失和梯度
            logit_s, logit_f, fs, ff = model(x)
            loss_ce_s = lam * F.cross_entropy(logit_s, y_c_a) + (1 - lam) * F.cross_entropy(logit_s, y_c_b)
            loss_ce_f = lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)
            loss = loss_ce_f + loss_ce_s
            
            if not args.no_reg:
                loss_kl = model.hier_kl(logit_f, logit_s, lambda_kl)
                loss_orth = model.orth_loss(fs, ff, args.lambda_orth)
                loss = loss + loss_kl + loss_orth
            
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            # 普通优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 4. 更新指标
        acc_s.update( (logit_s.argmax(1) == y_c_a).float().mean().item(), x.size(0) )
        acc_f.update( (logit_f.argmax(1) == y_a).float().mean().item(), x.size(0) )
        losses.update( loss.item(), x.size(0) )

        # 更新进度条
        pbar.set_postfix_str(f"loss={losses.val:.4f} (avg={losses.avg:.4f}) acc_s={acc_s.avg:.4f} acc_f={acc_f.avg:.4f}")

    # 记录到TensorBoard
    writer.add_scalar('Train/Loss', losses.avg, epoch)
    writer.add_scalar('Train/Accuracy_Super', acc_s.avg, epoch)
    writer.add_scalar('Train/Accuracy_Fine', acc_f.avg, epoch)
    if not args.no_reg:
        writer.add_scalar('Train/Loss_KL', loss_kl.item(), epoch)
        writer.add_scalar('Train/Loss_Orth', loss_orth.item(), epoch)
        writer.add_scalar('Train/Lambda_KL', lambda_kl, epoch)
    
    logger.info(f"Train E{epoch}: loss={losses.avg:.4f}  acc_s={acc_s.avg:.4f}  acc_f={acc_f.avg:.4f}")

@torch.no_grad()
def test(model, loader, epoch, writer):
    model.eval()
    correct_s = correct_f = total = 0
    pbar = tqdm(loader, total=len(loader), leave=False, desc=f"Test E{epoch}")

    for x, y in pbar:
        x, y = x.cuda(), y.cuda()
        y_c = model.f2c[y]
        
        logit_s, logit_f, _, _ = model(x)
        correct_s += (logit_s.argmax(1) == y_c).sum().item()
        correct_f += (logit_f.argmax(1) == y).sum().item()
        total += x.size(0)
        
    acc_s, acc_f = correct_s / total, correct_f / total
    
    # 记录到TensorBoard
    writer.add_scalar('Test/Accuracy_Super', acc_s, epoch)
    writer.add_scalar('Test/Accuracy_Fine', acc_f, epoch)
    
    logger.info(f"Test  E{epoch}: acc_s={acc_s:.4f}  acc_f={acc_f:.4f}")
    return acc_s, acc_f

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ---------- 主函数 ----------
def main():
    set_seed(42)
    
    # 获取数据加载器
    train_loader, test_loader = get_loader(args.batch, num_workers=args.num_workers, use_cutmix=args.use_cutmix)
    
    # 创建模型
    model = WideResNet(depth=16, widen=8, dropout=args.drop).cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # 获取优化器和调度器
    if args.use_sam:
        base_opt = optim.SGD
        optimizer = SAM(
            model.parameters(), 
            base_opt, 
            lr=args.lr, 
            momentum=0.9, 
            weight_decay=args.wd
        )
    else:
        # 使用AdamW优化器
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.wd
        )
    
    # 余弦退火调度器
    def sche(e): return 0.5 * (1 + math.cos(math.pi * e / args.epoch))
    # 适配不同的优化器
    if args.use_sam:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, sche)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, sche)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=args.save_dir)
    
    best_acc = 0.0
    start_epoch = 1
    
    # 加载检查点（如果提供）
    if args.resume:
        from pathlib import Path
        if Path(args.resume).exists():
            try:
                checkpoint = torch.load(args.resume)
                # 检查checkpoint字典中是否包含所有必要的键
                required_keys = ['model', 'optimizer', 'scheduler', 'epoch', 'best_acc']
                for key in required_keys:
                    if key not in checkpoint:
                        logger.error(f"❌ 检查点文件缺少必要的键: {key}")
                        exit(1)
                
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint['best_acc']
                
                logger.info(f"✅ 加载模型状态从: {args.resume}")
                logger.info(f"✅ 加载优化器状态")
                logger.info(f"✅ 加载调度器状态")
                logger.info(f"✅ 加载当前epoch: {checkpoint['epoch']}, 将从epoch {start_epoch} 开始")
                logger.info(f"✅ 加载最佳准确率: {best_acc:.4f}")
                logger.info(f"✅ 检查点加载成功: {args.resume}")
            except Exception as e:
                logger.error(f"❌ 加载检查点时出错: {e}")
                exit(1)
        else:
            logger.error(f"❌ 检查点文件不存在: {args.resume}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epoch + 1):
        if epoch <= args.warm:   # linear warm-up
            if args.use_sam:
                lr = args.lr * epoch / args.warm
                for param_group in optimizer.base_optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                lr = args.lr * epoch / args.warm
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
        else:
            scheduler.step()
        
        # 训练一个epoch
        train_one_epoch(model, train_loader, optimizer, epoch, writer, use_cutmix=args.use_cutmix)
        
        # 测试
        acc_s, acc_f = test(model, test_loader, epoch, writer)
        
        # 更新最佳准确率
        if acc_f > best_acc:
            best_acc = acc_f
            logger.info(f"*** Best model updated: {best_acc:.4f} ***")
            
            # 保存模型检查点
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, Path(args.save_dir) / f"best_model_{acc_f:.4f}.pth")
    
    logger.info(f"All done!")
    logger.info(f"Best accuracy: {best_acc:.4f}")
    writer.close()

if __name__ == "__main__":
    set_seed(42)
    main()