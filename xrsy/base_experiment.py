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
from tqdm import tqdm
from rich.logging import RichHandler
import logging
import math
import random, numpy as np
import pickle
import csv
import os

# ----------- 日志 -----------
def get_logger(experiment_name, log_file="train.log"):
    """创建带有实验名称前缀的日志记录器"""
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # 清除已存在的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(f"%(asctime)s | {experiment_name} | %(levelname)s | %(message)s")
    console_handler.setFormatter(console_formatter)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(f"%(asctime)s | {experiment_name} | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# ---------- 工具函数 ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- 数据处理 ----------
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# 获取细粒度到粗粒度的映射
train_path = Path("./data/cifar-100-python/train")
with train_path.open("rb") as f:
    train_data = pickle.load(f, encoding="bytes")

# 获取细粒度和粗粒度标签
fine_labels = train_data[b'fine_labels']
coarse_labels = train_data[b'coarse_labels']

# 创建细粒度到粗粒度的映射
fine2coarse_id = torch.zeros(100, dtype=torch.long)
for fine, coarse in zip(fine_labels, coarse_labels):
    fine2coarse_id[fine] = coarse

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

# ---------- WideResNet 模型 ----------
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
    def __init__(self, depth=16, widen=8, nc_coarse=20, nc_fine=100, dropout=0.0, use_coarse_branch=True, use_cross_attention=True, use_kl=True, attention_type='cross_branch'):
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
        self.use_coarse_branch = use_coarse_branch
        self.use_cross_attention = use_cross_attention
        self.use_kl = use_kl
        self.attention_type = attention_type  # 新增注意力类型参数
        
        # 特征提取层
        if use_coarse_branch:
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
            nn.Dropout(dropout),  # 使用统一的dropout率
            nn.Conv2d(nStages[3], nStages[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(nStages[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 使用统一的dropout率
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 定义跨分支注意力机制类
        class CrossBranchAttention(nn.Module):
            def __init__(self, embed_dim, dropout=0.0):
                super().__init__()
                self.embed_dim = embed_dim
                self.scale = embed_dim ** -0.5
                
                # 线性变换层 - 查询来自当前分支，键值来自另一个分支
                self.q_proj = nn.Linear(embed_dim, embed_dim)
                self.k_proj = nn.Linear(embed_dim, embed_dim)
                self.v_proj = nn.Linear(embed_dim, embed_dim)
                self.out_proj = nn.Linear(embed_dim, embed_dim)
                
                # 层归一化
                self.norm_query = nn.LayerNorm(embed_dim)
                self.norm_key_value = nn.LayerNorm(embed_dim)
                self.norm_output = nn.LayerNorm(embed_dim)  # 输出层归一化
                
                # 残差连接（直接使用Identity）
                self.residual = nn.Identity()
                
                self.dropout = nn.Dropout(dropout)
                self.softmax = nn.Softmax(dim=-1)
            
            def forward(self, query, key_value):
                # query: [batch_size, embed_dim] - 当前分支的特征
                # key_value: [batch_size, embed_dim] - 另一个分支的特征
                
                # 保存原始query用于残差连接
                original_query = query
                
                # 输入归一化
                query = self.norm_query(query)
                key_value = self.norm_key_value(key_value)
                
                # 线性变换
                q = self.q_proj(query).unsqueeze(1)  # [batch_size, 1, embed_dim]
                k = self.k_proj(key_value).unsqueeze(2)  # [batch_size, embed_dim, 1]
                v = self.v_proj(key_value).unsqueeze(1)  # [batch_size, 1, embed_dim]
                
                # 计算注意力分数 - 点积注意力
                # [batch_size, 1, embed_dim] x [batch_size, embed_dim, 1] -> [batch_size, 1, 1]
                attn_scores = torch.bmm(q, k) * self.scale
                attn_weights = self.softmax(attn_scores)  # [batch_size, 1, 1]
                
                # 应用注意力权重到值向量
                attn_output = torch.bmm(attn_weights, v).squeeze(1)  # [batch_size, embed_dim]
                attn_output = self.dropout(attn_output)  # 使用dropout防止过拟合
                
                # 最终线性变换和残差连接
                output = self.out_proj(attn_output)
                output = output + self.residual(original_query)  # 残差连接到原始输入
                output = self.norm_output(output)  # 输出层归一化
                
                return output
        
        if use_cross_attention:
            attention_type = getattr(self, 'attention_type', 'cross_branch')
            
            if attention_type == 'se':
                # SE（通道注意力）
                class SEBlock(nn.Module):
                    def __init__(self, channel, reduction=16):
                        super().__init__()
                        self.avg_pool = nn.AdaptiveAvgPool2d(1)
                        self.fc = nn.Sequential(
                            nn.Linear(channel, channel // reduction, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Linear(channel // reduction, channel, bias=False),
                            nn.Sigmoid()
                        )
                    
                    def forward(self, x):
                        b, c, _, _ = x.size()
                        y = self.avg_pool(x).view(b, c)
                        y = self.fc(y).view(b, c, 1, 1)
                        return y
                
                # 在forward方法中应用SE注意力
                self.se_super = SEBlock(nStages[3])
                self.se_fine = SEBlock(nStages[3])
                
                # 注意力模块设置为None，在forward中单独处理
                self.attention_super = None
                self.attention_fine = None
            elif attention_type == 'cross_branch':
                # 自定义CrossBranchAttention模块，处理两个输入张量
                class CrossBranchAttentionModule(nn.Module):
                    def __init__(self, embed_dim, dropout=0.0):
                        super().__init__()
                        self.attention = CrossBranchAttention(embed_dim, dropout)
                        # 移除额外的线性层和层归一化，因为注意力模块内部已经有了
                        # 简化设计以提高稳定性
                    
                    def forward(self, query, key_value):
                        out = self.attention(query, key_value)
                        return out
                
                # 添加分支间的交互机制 - 使用跨分支注意力机制
                self.attention_super = CrossBranchAttentionModule(nStages[3], dropout=dropout)
                self.attention_fine = CrossBranchAttentionModule(nStages[3], dropout=dropout)
            elif attention_type == 'none':
                # 无注意力
                self.attention_super = None
                self.attention_fine = None
        else:
            # 简单的线性注意力（用于消融实验）
            self.attention_super = nn.Sequential(
                nn.Linear(nStages[3], nStages[3] // 4),
                nn.ReLU(inplace=True),
                nn.Linear(nStages[3] // 4, nStages[3]),
                nn.Sigmoid()
            )
            
            self.attention_fine = nn.Sequential(
                nn.Linear(nStages[3], nStages[3] // 4),
                nn.ReLU(inplace=True),
                nn.Linear(nStages[3] // 4, nStages[3]),
                nn.Sigmoid()
            )

        # 分类头
        if use_coarse_branch:
            self.super_cls = nn.Linear(nStages[3], nc_coarse)
        self.fine_cls = nn.Linear(nStages[3], nc_fine)

        self.register_buffer("f2c", fine2coarse_id.detach().clone())

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

        # 分支分离的特征提取
        if self.use_coarse_branch:
            # 检查是否使用SE注意力
            if hasattr(self, 'se_super'):
                # SE注意力
                se_weight = self.se_super(out)
                out_s = out * se_weight
                fs_raw = self.super_branch(out_s)
            else:
                # 普通分支
                fs_raw = self.super_branch(out)
        
        if hasattr(self, 'se_fine'):
            # SE注意力
            se_weight = self.se_fine(out)
            out_f = out * se_weight
            ff_raw = self.fine_branch(out_f)
        else:
            # 普通分支
            ff_raw = self.fine_branch(out)
        
        if self.use_coarse_branch and self.use_cross_attention and self.attention_super is not None:
            # 分支间的注意力交互 - 改进的跨分支注意力
            # 超类分支使用细粒度分支的特征作为引导（query来自超类，key/value来自细粒度）
            fs_attended = self.attention_super(fs_raw, ff_raw)
            
            # 细粒度分支使用超类分支的特征作为引导（query来自细粒度，key/value来自超类）
            ff_attended = self.attention_fine(ff_raw, fs_raw)
        elif self.use_coarse_branch:
            # 无注意力交互
            fs_attended = fs_raw
            ff_attended = ff_raw
        else:
            # 无粗粒度分支
            ff_attended = ff_raw
        
        # 生成最终的分类logits
        if self.use_coarse_branch:
            logit_s = self.super_cls(fs_attended)
        else:
            logit_s = None
        logit_f = self.fine_cls(ff_attended)
        
        if self.use_coarse_branch:
            return logit_s, logit_f, fs_attended, ff_attended
        else:
            return logit_f

    def hier_kl(self, logit_f, logit_c, lam):
        prob_f = F.softmax(logit_f, 1)
        # 使用更高效的方式将细粒度概率映射到粗粒度概率
        batch_size = logit_f.size(0)
        nc_fine = prob_f.size(1)
        nc_coarse = logit_c.size(1)
        
        # 获取每个细粒度类别对应的粗粒度类别索引
        coarse_indices = self.f2c.to(prob_f.device)
        
        # 创建一个映射矩阵：nc_fine × nc_coarse
        mapping_matrix = torch.zeros(nc_fine, nc_coarse, device=prob_f.device)
        mapping_matrix[torch.arange(nc_fine), coarse_indices] = 1.0
        
        # 将细粒度概率分布转换为粗粒度概率分布
        prob_c = torch.matmul(prob_f, mapping_matrix)
        
        # 计算KL散度：粗粒度预测分布与细粒度映射到粗粒度的分布之间的差异
        # 注意：kl_div的第一个参数是log_softmax(预测), 第二个参数是目标分布
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

    @torch.no_grad()
    def first_step(self, zero_grad=False):
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
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        return torch.norm(torch.stack([p.grad.norm(p=2) for group in self.param_groups
                                      for p in group["params"] if p.grad is not None]), p=2)

# ---------- 平均指标计算器 ----------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    @property
    def avg(self): return self.sum / self.count

# ---------- 基础实验类 ----------
class BaseExperiment:
    def __init__(self, experiment_name, args):
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
            base_opt = optim.SGD
            optimizer = SAM(
                model.parameters(), 
                base_opt, 
                lr=self.args.lr, 
                momentum=0.9, 
                weight_decay=self.args.wd
            )
        else:
            # 使用AdamW优化器
            optimizer = optim.AdamW(
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
            scheduler = optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, sche)
        else:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, sche)
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
            model = nn.DataParallel(model)
        
        # 获取优化器和调度器
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)
        
        best_acc = 0.0
        start_epoch = 1
        
        # 加载检查点（如果提供）
        if self.args.resume:
            from pathlib import Path
            if Path(self.args.resume).exists():
                try:
                    checkpoint = torch.load(self.args.resume)
                    # 检查checkpoint字典中是否包含所有必要的键
                    required_keys = ['model', 'optimizer', 'scheduler', 'epoch', 'best_acc']
                    for key in required_keys:
                        if key not in checkpoint:
                            self.logger.error(f"❌ 检查点文件缺少必要的键: {key}")
                            exit(1)
                    
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    start_epoch = checkpoint['epoch'] + 1
                    best_acc = checkpoint['best_acc']
                    
                    self.logger.info(f"✅ 加载模型状态从: {self.args.resume}")
                    self.logger.info(f"✅ 加载优化器状态")
                    self.logger.info(f"✅ 加载调度器状态")
                    self.logger.info(f"✅ 加载当前epoch: {checkpoint['epoch']}, 将从epoch {start_epoch} 开始")
                    self.logger.info(f"✅ 加载最佳准确率: {best_acc:.4f}")
                    self.logger.info(f"✅ 检查点加载成功: {self.args.resume}")
                except Exception as e:
                    self.logger.error(f"❌ 加载检查点时出错: {e}")
                    exit(1)
            else:
                self.logger.error(f"❌ 检查点文件不存在: {self.args.resume}")
        
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

# ---------- 命令行参数解析 ----------
def parse_args():
    parser = argparse.ArgumentParser("Hierarchical Disentangled CIFAR-100 消融实验")
    parser.add_argument("--epoch", type=int, default=400, help="Total number of epochs to train")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--drop", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lambda_kl", type=float, default=0.2, help="层次KL散度损失的权重")
    parser.add_argument("--lambda_orth", type=float, default=5e-4, help="正交损失的权重")
    parser.add_argument("--warm", type=int, default=10, help="线性预热的epoch数")
    parser.add_argument("--no_reg", action="store_true", help="不使用正则化损失")
    parser.add_argument("--save_dir", type=str, default="ckpt", help="保存检查点的目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点文件路径")
    parser.add_argument("--num_workers", type=int, default=6, help="数据加载器的工作进程数")
    parser.add_argument("--use_cutmix", type=lambda x: x.lower() == 'true', default=True, help="是否使用CutMix数据增强")
    parser.add_argument("--strong_aug", action="store_true", default=False, help="是否使用更强的数据增强")
    
    # 消融实验参数
    parser.add_argument("--depth", type=int, default=16, help="WideResNet depth")
    parser.add_argument("--widen", type=int, default=8, help="WideResNet widen factor")
    parser.add_argument("--use_coarse_branch", type=bool, default=True, help="是否使用粗粒度分支")
    parser.add_argument("--use_cross_attention", type=bool, default=True, help="是否使用跨分支注意力")
    parser.add_argument("--use_kl", type=bool, default=True, help="是否使用层次KL散度")
    parser.add_argument("--use_dynamic_kl", type=bool, default=True, help="是否使用动态KL权重")
    parser.add_argument("--kl_warm_epochs", type=int, default=20, help="KL损失权重的warm-up epoch数")
    parser.add_argument("--kl_strategy", type=str, default="linear", choices=["linear", "cosine", "exponential", "logistic"], help="KL损失权重的动态调整策略")
    parser.add_argument("--kl_power", type=float, default=1.0, help="KL损失权重的动态调整指数（仅用于exponential策略）")
    parser.add_argument("--use_sam", type=bool, default=True, help="是否使用SAM优化器")
    # 注意力机制消融参数
    parser.add_argument("--attention_type", type=str, default="cross_branch", choices=["none", "se", "cross_branch"], help="注意力类型（none: 无注意力, se: SE通道注意力, cross_branch: 跨分支注意力）")
    
    return parser.parse_args()