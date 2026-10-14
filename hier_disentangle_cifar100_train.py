#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import RandAugment, RandomErasing
from torchvision import datasets, transforms
from tqdm import tqdm
import logging
import math

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

# ---------- 参 ----------
def parse_args():
    parser = argparse.ArgumentParser("WRN-28-10 + SAM  CIFAR-100")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--lambda_kl", type=float, default=0.2)
    parser.add_argument("--lambda_orth", type=float, default=5e-4)
    parser.add_argument("--warm", type=int, default=5)
    parser.add_argument("--no_reg", action="store_true")
    parser.add_argument("--save_dir", type=str, default="ckpt")
    return parser.parse_args()

args = parse_args()
Path(args.save_dir).mkdir(exist_ok=True)

# ---------- 数据 ----------
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

cutmix_beta = 1.0

def cutmix_collate(batch):
    """返回 CutMix 后的 tensor + 两个 target + lam"""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    lam = torch.distributions.Beta(cutmix_beta, cutmix_beta).sample().item()
    bsz = images.size(0)
    index = torch.randperm(bsz)
    W, H = images.size(-1), images.size(-2)
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    return images, labels, labels[index], lam

def get_loader(batch_size):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        RandomErasing(p=0.5, scale=(0.02, 0.15))
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    # 本地已有数据，download=False
    train_set = datasets.CIFAR100(root="./data/cifar-100-python", train=True,  download=False, transform=train_tf)
    test_set = datasets.CIFAR100(root="./data/cifar-100-python", train=False, download=False, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=6, pin_memory=True, drop_last=True,
                              collate_fn=cutmix_collate)
    test_loader = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=6, pin_memory=True)
    return train_loader, test_loader

# ---------- WRN-28-10 ----------
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
    def __init__(self, depth=28, widen=10, nc_coarse=20, nc_fine=100, dropout=0.0):
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

        # super branch
        self.super_dec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.super_cls = nn.Linear(nStages[3], nc_coarse)

        # fine branch
        self.fine_dec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fine_cls = nn.Linear(nStages[3], nc_fine)

        self.register_buffer("f2c", torch.tensor(
            [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 9, 4,
             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=torch.long))

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
        fs = self.super_dec(feat)
        logit_s = self.super_cls(fs)
        ff = self.fine_dec(feat)
        logit_f = self.fine_cls(ff)
        return logit_s, logit_f, fs, ff

    def hier_kl(self, logit_f, logit_c, lam):
        prob_f = F.softmax(logit_f, 1)
        prob_c = torch.zeros_like(logit_c)
        for fi in range(100):
            ci = self.f2c[fi]
            prob_c[:, ci] += prob_f[:, fi]
        return lam * F.kl_div(F.log_softmax(logit_c, 1), prob_c, reduction='batchmean')

    def orth_loss(self, fs, ff, lam):
        fs = F.normalize(fs, p=2, dim=1)
        ff = F.normalize(ff, p=2, dim=1)
        return lam * torch.abs((fs * ff).sum(1)).mean()

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

# ---------- 训练 ----------
def train_one_epoch(model, loader, optimizer, epoch):
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
            if args.no_reg:
                return loss_ce_s + loss_ce_f
            loss_kl   = model.hier_kl(logit_f, logit_s, args.lambda_kl)
            loss_orth = model.orth_loss(fs, ff, args.lambda_orth)
            return loss_ce_s + loss_ce_f + loss_kl + loss_orth

        loss = closure()
        optimizer.zero_grad()
        loss.backward()          # 先算一次梯度

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

    logger.info(f"Train E{epoch}: loss={losses.avg:.4f}  acc_s={acc_s.avg:.4f}  acc_f={acc_f.avg:.4f}")

@torch.no_grad()
def test(model, loader, epoch):
    model.eval()
    correct_s = correct_f = total = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        y_c = model.f2c[y]
        logit_s, logit_f, _, _ = model(x)
        correct_s += (logit_s.argmax(1) == y_c).sum().item()
        correct_f += (logit_f.argmax(1) == y).sum().item()
        total += x.size(0)
    acc_s, acc_f = correct_s / total, correct_f / total
    logger.info(f"Test  E{epoch}: acc_s={acc_s:.4f}  acc_f={acc_f:.4f}")
    return acc_f

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    @property
    def avg(self): return self.sum / self.count

# ---------- 主 ----------
def main():
    train_loader, test_loader = get_loader(args.batch)
    model = WideResNet(depth=28, widen=10, dropout=args.drop).cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    base_opt = optim.SGD
    optimizer = SAM(model.parameters(), base_opt, lr=args.lr, momentum=0.9, weight_decay=args.wd)

    # WD cosine 退火
    def sche(e): return 0.5 * (1 + math.cos(math.pi * e / args.epoch))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, sche)

    best_acc = 0.0
    for epoch in range(1, args.epoch + 1):
        if epoch <= args.warm:   # linear warm-up
            for g in optimizer.param_groups:
                g["lr"] = args.lr * epoch / args.warm
        else:
            scheduler.step(epoch - args.warm)

        train_one_epoch(model, train_loader, optimizer, epoch)
        acc_f = test(model, test_loader, epoch)

        is_best = acc_f > best_acc
        best_acc = max(best_acc, acc_f)
        state = {"model": model.state_dict(), "epoch": epoch, "best_acc": best_acc}
        torch.save(state, f"{args.save_dir}/latest.pth")
        if is_best:
            torch.save(state, f"{args.save_dir}/best.pth")
            logger.info(f"*** Best model updated: {best_acc:.4f} ***")

    logger.info("All done!")

if __name__ == "__main__":
    main()