
"""
======================================================================
Stanford Cars Hierarchical Baseline
- Fine-grained branch: 196 car classes
- Coarse-grained branch: brand-level classes parsed from class names
- Supports CutMix / KL / orthogonality / cross-branch attention / SAM
======================================================================
"""

import argparse
import csv
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import RandomErasing, AutoAugment, AutoAugmentPolicy
from tqdm import tqdm


# =====================
# Global config
# =====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
num_epochs = 300
train_ratios = [0.5, 1.0]
log_file = "stanford_cars_baseline_coarse.log"
use_cutmix = False
cutmix_beta = 1.0
seed = 42
img_size = 128

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed_value: int = 42) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(experiment_name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


# =====================
# Dataset helpers
# =====================
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    AutoAugment(AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    RandomErasing(p=0.25, scale=(0.02, 0.10)),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def cutmix_collate(batch):
    """CutMix for arbitrary input size."""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)

    lam = np.random.beta(cutmix_beta, cutmix_beta)
    bsz = images.size(0)
    index = torch.randperm(bsz)

    H = images.size(2)
    W = images.size(3)
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    return images, labels, labels[index], lam


def get_stanford_cars(root='./data'):
    """
    下载并准备Stanford Cars数据集
    """
    # 使用torchvision的StanfordCars数据集（需要最新版torchvision）
    try:
        train_dataset = datasets.StanfordCars(
            root=root,
            split='train',
            download=True,
            transform=transform_train
        )
        test_dataset = datasets.StanfordCars(
            root=root,
            split='test',
            download=True,
            transform=transform_test
        )
        return train_dataset, test_dataset
    except:

        # 使用ImageFolder作为备选（需要手动组织数据）
        train_dataset = datasets.StanfordCars(
            root='./data',
            split='train',
            download=False,
            transform=transform_train
        )
        test_dataset = datasets.StanfordCars(
            root='./data',
            split='test',
            download=False,
            transform=transform_train
        )
        return train_dataset, test_dataset


def extract_brand_from_class_name(class_name: str) -> str:
    """
    Parse brand/make from Stanford Cars class name.
    Example:
        'AM General Hummer SUV 2000' -> 'AM General'
        'Aston Martin V8 Vantage Coupe 2012' -> 'Aston Martin'
        'Audi TTS Coupe 2012' -> 'Audi'
    """
    multi_word_brands = [
        'AM General',
        'Aston Martin',
        'Land Rover',
        'Rolls-Royce',
    ]

    for brand in multi_word_brands:
        if class_name.startswith(brand + ' '):
            return brand

    return class_name.split(' ')[0]


class StanfordCarsHierarchy:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.fine_to_brand_name = [extract_brand_from_class_name(name) for name in class_names]
        self.brand_names = sorted(set(self.fine_to_brand_name))
        self.brand_to_idx = {brand: idx for idx, brand in enumerate(self.brand_names)}
        self.fine2coarse = torch.tensor(
            [self.brand_to_idx[b] for b in self.fine_to_brand_name],
            dtype=torch.long
        )

    @property
    def num_fine(self) -> int:
        return len(self.class_names)

    @property
    def num_coarse(self) -> int:
        return len(self.brand_names)


def build_hierarchy_from_dataset(dataset) -> StanfordCarsHierarchy:
    class_names = None

    if hasattr(dataset, 'classes') and dataset.classes is not None:
        class_names = list(dataset.classes)
    elif hasattr(dataset, '_samples') and dataset._samples is not None:
        # Fallback path if API changes
        class_names = sorted({s[0] for s in dataset._samples})

    if class_names is None:
        raise RuntimeError('Unable to retrieve class names from StanfordCars dataset.')

    return StanfordCarsHierarchy(class_names)


# =====================
# Model blocks
# =====================
class WideBasic(nn.Module):
    def __init__(self, c_in, c_out, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride, 1, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Conv2d(c_in, c_out, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.drop(out)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return out


class CrossBranchAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        residual = query
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)

        q = self.q_proj(query).unsqueeze(1)     # [B,1,C]
        k = self.k_proj(key_value).unsqueeze(2) # [B,C,1]
        v = self.v_proj(key_value).unsqueeze(1) # [B,1,C]

        attn = torch.bmm(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).squeeze(1)
        out = self.dropout(out)
        out = self.out_proj(out)
        out = self.norm_out(out + residual)
        return out


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        hidden = max(channel // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class WideResNet(nn.Module):
    def __init__(
        self,
        depth=16,
        widen=8,
        nc_coarse=49,
        nc_fine=196,
        dropout=0.0,
        use_coarse_branch=True,
        use_cross_attention=True,
        use_kl=True,
        attention_type='cross_branch',
        fine2coarse_id: torch.Tensor = None,
    ):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must be of the form 6n+4, e.g. 16, 22, 28, 40")

        n = (depth - 4) // 6
        k = widen
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.use_coarse_branch = use_coarse_branch
        self.use_cross_attention = use_cross_attention
        self.use_kl = use_kl
        self.attention_type = attention_type
        self.feature_dim = nStages[3]

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(WideBasic, nStages[1], n, 1, dropout)
        self.layer2 = self._make_layer(WideBasic, nStages[2], n, 2, dropout)
        self.layer3 = self._make_layer(WideBasic, nStages[3], n, 2, dropout)
        self.bn = nn.BatchNorm2d(nStages[3])

        if self.use_coarse_branch:
            self.super_branch = nn.Sequential(
                nn.Conv2d(nStages[3], nStages[3], 5, 1, 2, bias=False),
                nn.BatchNorm2d(nStages[3]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        self.fine_branch = nn.Sequential(
            nn.Conv2d(nStages[3], nStages[3], 1, 1, 0, bias=False),
            nn.BatchNorm2d(nStages[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(nStages[3], nStages[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(nStages[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.se_super = None
        self.se_fine = None
        self.attention_super = None
        self.attention_fine = None

        if self.use_cross_attention:
            if attention_type == 'se':
                if self.use_coarse_branch:
                    self.se_super = SEBlock(nStages[3])
                self.se_fine = SEBlock(nStages[3])
            elif attention_type == 'cross_branch':
                if self.use_coarse_branch:
                    self.attention_super = CrossBranchAttention(nStages[3], dropout)
                    self.attention_fine = CrossBranchAttention(nStages[3], dropout)
            elif attention_type == 'none':
                pass
            else:
                raise ValueError(f'Unsupported attention_type: {attention_type}')

        if self.use_coarse_branch:
            self.super_cls = nn.Linear(nStages[3], nc_coarse)
        self.fine_cls = nn.Linear(nStages[3], nc_fine)

        if fine2coarse_id is not None:
            self.register_buffer('f2c', fine2coarse_id.detach().clone().long())
        else:
            self.f2c = None

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
        out = F.relu(self.bn(out), inplace=True)

        if self.use_coarse_branch:
            out_s = self.se_super(out) if self.se_super is not None else out
            fs_raw = self.super_branch(out_s)
        else:
            fs_raw = None

        out_f = self.se_fine(out) if self.se_fine is not None else out
        ff_raw = self.fine_branch(out_f)

        if self.use_coarse_branch:
            if self.attention_super is not None and self.attention_fine is not None:
                fs = self.attention_super(fs_raw, ff_raw)
                ff = self.attention_fine(ff_raw, fs_raw)
            else:
                fs = fs_raw
                ff = ff_raw

            logit_s = self.super_cls(fs)
            logit_f = self.fine_cls(ff)
            return logit_s, logit_f, fs, ff

        logit_f = self.fine_cls(ff_raw)
        return logit_f

    def hier_kl(self, logit_f: torch.Tensor, logit_c: torch.Tensor, lam: float) -> torch.Tensor:
        if self.f2c is None:
            return torch.tensor(0.0, device=logit_f.device)

        prob_f = F.softmax(logit_f, dim=1)
        nc_fine = prob_f.size(1)
        nc_coarse = logit_c.size(1)
        coarse_indices = self.f2c.to(prob_f.device)

        mapping = torch.zeros(nc_fine, nc_coarse, device=prob_f.device)
        mapping[torch.arange(nc_fine, device=prob_f.device), coarse_indices] = 1.0
        prob_c_from_f = torch.matmul(prob_f, mapping)
        loss_kl = F.kl_div(F.log_softmax(logit_c, dim=1), prob_c_from_f, reduction='batchmean')
        return lam * loss_kl

    def orth_loss(self, fs: torch.Tensor, ff: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Stable orthogonality regularization.
        - inter-branch cosine similarity penalty
        - batch Gram decorrelation penalty
        """
        fs = F.normalize(fs, p=2, dim=1)
        ff = F.normalize(ff, p=2, dim=1)

        loss_inter = torch.abs((fs * ff).sum(dim=1)).mean()

        bs = fs.size(0)
        if bs > 1:
            gram_s = torch.matmul(fs, fs.t())
            gram_f = torch.matmul(ff, ff.t())
            eye = torch.eye(bs, device=fs.device)
            loss_gram_s = ((gram_s - eye) ** 2).mean()
            loss_gram_f = ((gram_f - eye) ** 2).mean()
        else:
            loss_gram_s = torch.tensor(0.0, device=fs.device)
            loss_gram_f = torch.tensor(0.0, device=fs.device)

        return lam * (loss_inter + loss_gram_s + loss_gram_f)


# =====================
# Optimizer helpers
# =====================
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# =====================
# Experiment
# =====================
class BaseExperiment:
    def __init__(self, experiment_name, args):
        self.experiment_name = experiment_name
        self.args = args
        self.device = torch.device(device)

        Path('cars_base_lines/logs').mkdir(exist_ok=True)
        Path('cars_base_lines/csv').mkdir(exist_ok=True)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(experiment_name, f'logs/{experiment_name}.log')
        self.csv_file = f'csv/{experiment_name}.csv'
        self.writer = SummaryWriter(log_dir=os.path.join(args.save_dir, experiment_name))

        set_seed(getattr(args, 'seed', 42))
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_acc_s', 'train_acc_f',
                'test_acc_s', 'test_acc_f', 'best_acc_f'
            ])

    def write_to_csv(self, epoch, train_loss, train_acc_s, train_acc_f, test_acc_s, test_acc_f, best_acc):
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc_s, train_acc_f, test_acc_s, test_acc_f, best_acc])

    def get_data_loader(self, batch_size, use_cutmix=False):
        train_set, test_set = get_stanford_cars(root='./data')
        hierarchy = build_hierarchy_from_dataset(train_set)
        self.logger.info(f'Fine classes: {hierarchy.num_fine}, coarse classes (brands): {hierarchy.num_coarse}')
        self.logger.info(f'Brand classes: {hierarchy.brand_names}')

        if hasattr(self.args, 'train_ratio') and self.args.train_ratio < 1.0:
            num_samples = int(len(train_set) * self.args.train_ratio)
            rng = np.random.default_rng(getattr(self.args, 'seed', 42))
            indices = rng.choice(len(train_set), num_samples, replace=False)
            train_set = Subset(train_set, indices.tolist())
            self.logger.info(f'Using train ratio={self.args.train_ratio:.1%}, sampled {len(train_set)} / {len(indices)}')

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=cutmix_collate if use_cutmix else None,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader, hierarchy

    def get_optimizer(self, model):
        if self.args.use_sam:
            return SAM(
                model.parameters(),
                optim.SGD,
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.wd,
            )
        return optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

    def get_scheduler(self, optimizer):
        def sche(e):
            return 0.5 * (1 + math.cos(math.pi * e / self.args.epoch))

        if self.args.use_sam:
            return optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, sche)
        return optim.lr_scheduler.LambdaLR(optimizer, sche)

    def _compute_lambda_kl(self, epoch: int) -> float:
        if not self.args.use_dynamic_kl:
            return self.args.lambda_kl

        if epoch <= self.args.kl_warm_epochs:
            t = epoch / max(self.args.kl_warm_epochs, 1)
            if self.args.kl_strategy == 'linear':
                return self.args.lambda_kl * t
            if self.args.kl_strategy == 'cosine':
                return self.args.lambda_kl * (1 - math.cos(math.pi * t)) / 2.0
            if self.args.kl_strategy == 'exponential':
                return self.args.lambda_kl * (t ** max(self.args.kl_power, 1e-6))
            if self.args.kl_strategy == 'logistic':
                return self.args.lambda_kl / (1 + math.exp(-10 * (t - 0.5)))
        return self.args.lambda_kl

    def _forward_loss(self, model, x, y_a, y_b, lam, lambda_kl):
        raw_model = unwrap_model(model)

        if raw_model.use_coarse_branch:
            y_c_a = raw_model.f2c[y_a]
            y_c_b = raw_model.f2c[y_b]
            logit_s, logit_f, fs, ff = model(x)

            loss_ce_s = lam * F.cross_entropy(logit_s, y_c_a) + (1 - lam) * F.cross_entropy(logit_s, y_c_b)
            loss_ce_f = lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)

            loss = loss_ce_s + loss_ce_f

            if not self.args.no_reg:
                if raw_model.use_kl:
                    loss += raw_model.hier_kl(logit_f, logit_s, lambda_kl)

                if getattr(self.args, "use_orth", True):
                    loss += raw_model.orth_loss(fs, ff, self.args.lambda_orth)

            return loss

        logit_f = model(x)
        return lam * F.cross_entropy(logit_f, y_a) + (1 - lam) * F.cross_entropy(logit_f, y_b)
    def train_one_epoch(self, model, loader, optimizer, epoch):
        raw_model = unwrap_model(model)
        model.train()
        losses, acc_s, acc_f = AverageMeter(), AverageMeter(), AverageMeter()
        pbar = tqdm(loader, total=len(loader), leave=False, desc=f'Epoch {epoch}')

        lambda_kl = self._compute_lambda_kl(epoch)

        for batch in pbar:
            if len(batch) == 2:
                x, y = batch
                y_a, y_b, lam = y, y, 1.0
            else:
                x, y_a, y_b, lam = batch

            x = x.to(self.device, non_blocking=True)
            y_a = y_a.to(self.device, non_blocking=True)
            y_b = y_b.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            loss = self._forward_loss(model, x, y_a, y_b, lam, lambda_kl)
            loss.backward()

            if self.args.use_sam:
                optimizer.first_step(zero_grad=True)
                second_loss = self._forward_loss(model, x, y_a, y_b, lam, lambda_kl)
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            with torch.no_grad():
                if raw_model.use_coarse_branch:
                    y_c_a = raw_model.f2c[y_a]
                    y_c_b = raw_model.f2c[y_b]
                    logit_s, logit_f, _, _ = model(x)
                    pred_s = logit_s.argmax(dim=1)
                    pred_f = logit_f.argmax(dim=1)
                    acc_s_batch = lam * (pred_s == y_c_a).float().mean() + (1 - lam) * (pred_s == y_c_b).float().mean()
                    acc_f_batch = lam * (pred_f == y_a).float().mean() + (1 - lam) * (pred_f == y_b).float().mean()
                    acc_s.update(acc_s_batch.item(), x.size(0))
                    acc_f.update(acc_f_batch.item(), x.size(0))
                else:
                    logit_f = model(x)
                    pred_f = logit_f.argmax(dim=1)
                    acc_f_batch = lam * (pred_f == y_a).float().mean() + (1 - lam) * (pred_f == y_b).float().mean()
                    acc_f.update(acc_f_batch.item(), x.size(0))

                losses.update(loss.item(), x.size(0))
                pbar.set_postfix(loss=f'{losses.avg:.4f}', acc_s=f'{acc_s.avg:.4f}', acc_f=f'{acc_f.avg:.4f}')

        self.logger.info(
            f'Train E{epoch}: loss={losses.avg:.4f} | acc_s={acc_s.avg:.4f} | acc_f={acc_f.avg:.4f} | lambda_kl={lambda_kl:.4f}'
        )
        self.writer.add_scalar('Train/Loss', losses.avg, epoch)
        self.writer.add_scalar('Train/Accuracy_Super', acc_s.avg, epoch)
        self.writer.add_scalar('Train/Accuracy_Fine', acc_f.avg, epoch)
        return losses.avg, acc_s.avg, acc_f.avg

    @torch.no_grad()
    def test(self, model, loader, epoch):
        raw_model = unwrap_model(model)
        model.eval()
        correct_s = 0
        correct_f = 0
        total = 0

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if raw_model.use_coarse_branch:
                y_c = raw_model.f2c[y]
                logit_s, logit_f, _, _ = model(x)
                correct_s += (logit_s.argmax(dim=1) == y_c).sum().item()
            else:
                logit_f = model(x)

            correct_f += (logit_f.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        acc_s = correct_s / total if raw_model.use_coarse_branch else 0.0
        acc_f = correct_f / total
        self.logger.info(f'Test  E{epoch}: acc_s={acc_s:.4f} | acc_f={acc_f:.4f}')
        self.writer.add_scalar('Test/Accuracy_Super', acc_s, epoch)
        self.writer.add_scalar('Test/Accuracy_Fine', acc_f, epoch)
        return acc_s, acc_f

    def run(self):
        self.logger.info(f'Starting experiment: {self.experiment_name}')
        self.logger.info(f'Arguments: {self.args}')
        self.logger.info(f'Device: {self.device}')

        train_loader, test_loader, hierarchy = self.get_data_loader(self.args.batch, use_cutmix=self.args.use_cutmix)

        model = WideResNet(
            depth=self.args.depth,
            widen=self.args.widen,
            nc_coarse=hierarchy.num_coarse,
            nc_fine=hierarchy.num_fine,
            dropout=self.args.drop,
            use_coarse_branch=self.args.use_coarse_branch,
            use_cross_attention=self.args.use_cross_attention,
            use_kl=self.args.use_kl,
            attention_type=self.args.attention_type,
            fine2coarse_id=hierarchy.fine2coarse,
        ).to(self.device)

        if torch.cuda.device_count() > 1:
            self.logger.info(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
            model = nn.DataParallel(model)

        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)

        best_acc = 0.0
        start_epoch = 1

        if self.args.resume and Path(self.args.resume).exists():
            checkpoint = torch.load(self.args.resume, map_location=self.device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            self.logger.info(f'Resumed from {self.args.resume} at epoch {start_epoch}, best_acc={best_acc:.4f}')

        for epoch in range(start_epoch, self.args.epoch + 1):
            if epoch <= self.args.warm:
                target_optimizer = optimizer.base_optimizer if self.args.use_sam else optimizer
                for g in target_optimizer.param_groups:
                    g['lr'] = self.args.lr * epoch / max(self.args.warm, 1)

            train_loss, train_acc_s, train_acc_f = self.train_one_epoch(model, train_loader, optimizer, epoch)

            if epoch > self.args.warm:
                scheduler.step()

            test_acc_s, test_acc_f = self.test(model, test_loader, epoch)
            is_best = test_acc_f > best_acc
            best_acc = max(best_acc, test_acc_f)

            self.write_to_csv(epoch, train_loss, train_acc_s, train_acc_f, test_acc_s, test_acc_f, best_acc)

            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }
            torch.save(state, os.path.join(self.args.save_dir, f'{self.experiment_name}_latest.pth'))
            if is_best:
                torch.save(state, os.path.join(self.args.save_dir, f'{self.experiment_name}_best.pth'))
                self.logger.info(f'*** Best model updated: {best_acc:.4f} ***')

        self.logger.info(f'Finished. Best fine accuracy: {best_acc:.4f}')
        self.writer.close()


# =====================
# Args and entry
# =====================
def parse_args():
    parser = argparse.ArgumentParser('Stanford Cars WRN-16-8 Baseline')
    parser.add_argument('--epoch', type=int, default=num_epochs)
    parser.add_argument('--batch', type=int, default=batch_size)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--drop', type=float, default=0.10)

    parser.add_argument('--lambda_kl', type=float, default=0.2)
    parser.add_argument('--lambda_orth', type=float, default=5e-4)

    parser.add_argument('--warm', type=int, default=10)
    parser.add_argument('--no_reg', action='store_true')
    parser.add_argument('--save_dir', type=str, default='car_checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--use_cutmix', type=lambda x: x.lower() == 'true', default=False)

    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--widen', type=int, default=8)

    # 只跑 WRN-16-8 baseline：关闭所有层级模块
    parser.add_argument('--use_coarse_branch', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_cross_attention', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_kl', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_dynamic_kl', type=lambda x: x.lower() == 'true', default=False)

    parser.add_argument('--kl_warm_epochs', type=int, default=20)
    parser.add_argument('--kl_strategy', type=str, default='linear',
                        choices=['linear', 'cosine', 'exponential', 'logistic'])
    parser.add_argument('--kl_power', type=float, default=1.0)

    parser.add_argument('--use_sam', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--attention_type', type=str, default='none',
                        choices=['none', 'se', 'cross_branch'])

    parser.add_argument('--train_ratio', type=float, default=1.0, choices=[0.5, 1.0])
    parser.add_argument('--seed', type=int, default=seed)
    return parser.parse_args()


def main():
    args = parse_args()
    exp_name = f'stanford_cars_ratio{int(args.train_ratio * 100)}_wrn{args.depth}-{args.widen}'
    experiment = BaseExperiment(exp_name, args)
    experiment.run()


if __name__ == '__main__':
    main()