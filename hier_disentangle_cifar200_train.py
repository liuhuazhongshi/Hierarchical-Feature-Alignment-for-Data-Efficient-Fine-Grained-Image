#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dual-branch WideResNet + Hierarchical Feature Enhancement (Dynamic KL + Attention)
CIFAR-100 Training Script
"""
import argparse, random, math, pickle, pathlib, logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment, RandomErasing
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import logging


logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s - %(message)s')


# ---------------- 随机种子 ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- 参数 ----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--lambda_kl", type=float, default=0.2)
    parser.add_argument("--warm", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="ckpt")
    parser.add_argument("--num_workers", type=int, default=6)
    return parser.parse_args()
args = parse_args()
Path(args.save_dir).mkdir(exist_ok=True)

# ---------------- 数据 ----------------
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
cutmix_beta = 1.0

# 加载 fine->coarse 映射
train_path = pathlib.Path("./data/cifar-100-python/train")
with train_path.open("rb") as f:
    train_data = pickle.load(f, encoding="bytes")
fine_labels = train_data[b'fine_labels']
coarse_labels = train_data[b'coarse_labels']
fine2coarse_id = torch.zeros(100, dtype=torch.long)
for f_lab, c_lab in zip(fine_labels, coarse_labels):
    fine2coarse_id[f_lab] = c_lab

def cutmix_collate(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    lam = torch.distributions.Beta(cutmix_beta, cutmix_beta).sample().item()
    bsz = images.size(0)
    index = torch.randperm(bsz)
    W, H = images.size(-1), images.size(-2)
    cut_rat = math.sqrt(1. - lam)
    cut_w, cut_h = max(1,int(W*cut_rat)), max(1,int(H*cut_rat))
    bbx1 = torch.randint(0, W - cut_w + 1, (1,)).item()
    bby1 = torch.randint(0, H - cut_h + 1, (1,)).item()
    bbx2, bby2 = bbx1+cut_w, bby1+cut_h
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - (bbx2-bbx1)*(bby2-bby1)/(W*H)
    return images, labels, labels[index], lam

def get_loader(batch_size, num_workers=6):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # 增加随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        RandAugment(num_ops=3, magnitude=9),  # 增强RandAugment
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))  # 随机擦除
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# ---------------- WRN + Dual Branch + Hierarchical Enhancement ----------------

class WideBasic(nn.Module):
    def __init__(self, c_in, c_out, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride, 1, bias=False)
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.shortcut = nn.Conv2d(c_in, c_out, 1, stride, bias=False) if stride!=1 or c_in!=c_out else nn.Identity()
    def forward(self,x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.drop(out)
        out = self.conv2(out)
        return out + self.shortcut(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, nc_coarse=20, nc_fine=100, dropout=0.0):
        super().__init__()
        nStages = [16, 16*widen, 32*widen, 64*widen]
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, nStages[0], 3,1,1,bias=False)
        n = (depth-4)//6
        self.layer1 = self._make_layer(WideBasic, nStages[1], n, 1, dropout)
        self.layer2 = self._make_layer(WideBasic, nStages[2], n, 2, dropout)
        self.layer3 = self._make_layer(WideBasic, nStages[3], n, 2, dropout)
        self.bn = nn.BatchNorm2d(nStages[3])

        # 超类分支
        self.super_branch = nn.Sequential(nn.Conv2d(nStages[3], nStages[3], 5,1,2,bias=False),
                                          nn.BatchNorm2d(nStages[3]), nn.ReLU(),
                                          nn.AdaptiveAvgPool2d(1), nn.Flatten())
        # 细粒度分支
        self.fine_branch = nn.Sequential(nn.Conv2d(nStages[3], nStages[3],1,1,0,bias=False),
                                         nn.BatchNorm2d(nStages[3]), nn.ReLU(),
                                         nn.Conv2d(nStages[3], nStages[3],3,1,1,bias=False),
                                         nn.BatchNorm2d(nStages[3]), nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(1), nn.Flatten())

        # 强化层次感知：cross-branch attention
        self.attention_super = nn.Sequential(nn.Linear(nStages[3], nStages[3]//4), nn.ReLU(),
                                             nn.Linear(nStages[3]//4, nStages[3]), nn.Sigmoid())
        self.attention_fine  = nn.Sequential(nn.Linear(nStages[3], nStages[3]//4), nn.ReLU(),
                                             nn.Linear(nStages[3]//4, nStages[3]), nn.Sigmoid())

        # 分类头
        self.super_cls = nn.Linear(nStages[3], nc_coarse)
        self.fine_cls  = nn.Linear(nStages[3], nc_fine)

        self.register_buffer("f2c", fine2coarse_id.detach().clone())

        # 初始化动态KL权重
        self.kl_base = 0.2

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        layers = [block(self.in_planes, planes, stride, dropout)]
        self.in_planes = planes
        for _ in range(1,num_blocks):
            layers.append(block(planes, planes, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self,x, epoch=None, max_epoch=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))

        fs = self.super_branch(out)
        ff = self.fine_branch(out)

        # cross-branch attention
        fs_att = fs * self.attention_super(ff)
        ff_att = ff * self.attention_fine(fs)

        # 分类logits
        logit_s = self.super_cls(fs_att)
        logit_f = self.fine_cls(ff_att)

        # 动态KL权重
        kl_weight = self.kl_base
        if epoch is not None and max_epoch is not None:
            kl_weight = self.kl_base * min(1.0, epoch/max_epoch)  # 训练前期小，后期大

        return logit_s, logit_f, fs_att, ff_att, kl_weight

    def hier_kl(self, logit_f, logit_c, lam):
        prob_f = F.softmax(logit_f,1)
        prob_c = torch.zeros_like(logit_c).scatter_add_(1, self.f2c.expand(prob_f.size(0),-1).to(prob_f.device), prob_f)
        return lam * F.kl_div(F.log_softmax(logit_c,1), prob_c, reduction='batchmean')

# ---------------- 优化器：SAM ----------------
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for g in self.param_groups for p in g['params'] if p.grad is not None]), p=2)
        for group in self.param_groups:
            scale = group['rho']/(grad_norm+1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad*scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

# ---------------- 工具类 ----------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.sum=self.count=0
    def update(self,val,n=1): self.val=val; self.sum+=val*n; self.count+=n
    @property
    def avg(self): return self.sum/self.count

# ---------------- 训练与测试 ----------------
def train_one_epoch(model, loader, optimizer, epoch, writer, max_epoch):
    model.train()
    losses, acc_s, acc_f = AverageMeter(), AverageMeter(), AverageMeter()
    pbar = tqdm(loader, total=len(loader), leave=False, desc=f"E{epoch}")
    for x, y in pbar:
        x, y = x.cuda(), y.cuda()
        y_a = y
        y_b = y
        lam = 1.0
        y_c_a, y_c_b = model.f2c[y_a], model.f2c[y_b]
        def closure():
            logit_s, logit_f, fs, ff, kl_weight = model(x, epoch, max_epoch)
            loss_s = lam*F.cross_entropy(logit_s, y_c_a) + (1-lam)*F.cross_entropy(logit_s,y_c_b)
            loss_f = lam*F.cross_entropy(logit_f, y_a) + (1-lam)*F.cross_entropy(logit_f,y_b)
            loss_kl = model.hier_kl(logit_f, logit_s, kl_weight)
            return loss_s + loss_f + loss_kl
        optimizer.zero_grad()
        loss = closure(); loss.backward()
        optimizer.first_step(zero_grad=False)
        loss = closure(); loss.backward()
        optimizer.second_step(zero_grad=True)
        with torch.no_grad():
            logit_s, logit_f, _, _, _ = model(x)
            pred_s, pred_f = logit_s.argmax(1), logit_f.argmax(1)
            acc_s.update((lam*(pred_s==y_c_a).float() + (1-lam)*(pred_s==y_c_b).float()).mean().item(), x.size(0))
            acc_f.update((lam*(pred_f==y_a).float() + (1-lam)*(pred_f==y_b).float()).mean().item(), x.size(0))
            losses.update(loss.item(), x.size(0))
        pbar.set_postfix(loss=losses.avg, acc_s=acc_s.avg, acc_f=acc_f.avg)
    writer.add_scalar('Train/Loss', losses.avg, epoch)
    writer.add_scalar('Train/Acc_Super', acc_s.avg, epoch)
    writer.add_scalar('Train/Acc_Fine', acc_f.avg, epoch)
    logging.info(
        f"Train E{epoch}: loss={losses.avg:.4f} "
        f"acc_s={acc_s.avg:.4f} acc_f={acc_f.avg:.4f}"
    )

@torch.no_grad()
def test(model, loader, epoch, writer):
    model.eval()
    correct_s=correct_f=total=0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        y_c = model.f2c[y]
        logit_s, logit_f, _, _, _ = model(x)
        correct_s += (logit_s.argmax(1)==y_c).sum().item()
        correct_f += (logit_f.argmax(1)==y).sum().item()
        total += x.size(0)
    acc_s, acc_f = correct_s/total, correct_f/total
    writer.add_scalar('Test/Acc_Super', acc_s, epoch)
    writer.add_scalar('Test/Acc_Fine', acc_f, epoch)
    logging.info(f"Test E{epoch}: acc_s={acc_s:.4f}  acc_f={acc_f:.4f}")
    return acc_s, acc_f

# ---------------- 主程序 ----------------
def main():
    train_loader, test_loader = get_loader(args.batch, args.num_workers)
    model = WideResNet(depth=16, widen=8,dropout=0.3).cuda()
    if torch.cuda.device_count()>1: model = nn.DataParallel(model)
    base_opt = optim.SGD
    optimizer = SAM(model.parameters(), base_opt, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=args.epoch)
    writer = SummaryWriter(log_dir=args.save_dir)
    best_acc=0.0
    for epoch in range(1, args.epoch+1):
        # Linear warmup
        if epoch<=args.warm:
            for g in optimizer.param_groups: g['lr'] = args.lr*epoch/args.warm
        train_one_epoch(model, train_loader, optimizer, epoch, writer, args.epoch)
        if epoch>args.warm: scheduler.step()
        acc_s, acc_f = test(model, test_loader, epoch, writer)
        is_best = acc_f>best_acc; best_acc=max(best_acc, acc_f)
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"best_acc":best_acc}
        torch.save(state, f"{args.save_dir}/latest.pth")
        if is_best: torch.save(state, f"{args.save_dir}/best.pth")
    logging.info("Training finished!")

if __name__=="__main__":
    set_seed(42)

    main()
