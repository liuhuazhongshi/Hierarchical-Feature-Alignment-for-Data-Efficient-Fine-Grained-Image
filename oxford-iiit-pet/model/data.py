#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理模块 - 增强版数据增强
"""
import math
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import RandomErasing, AutoAugment, AutoAugmentPolicy
from torchvision import datasets, transforms
import datasets as hf_datasets
import random
from PIL import Image, ImageFilter, ImageEnhance

# Oxford-IIIT Pet 均值和标准差（根据数据集统计）
OXFORD_IIIT_PET_MEAN = (0.485, 0.456, 0.406)
OXFORD_IIIT_PET_STD = (0.229, 0.224, 0.225)

# Oxford-IIIT Pet类别数量
NUM_CLASSES = 37
NUM_COARSE_CLASSES = 8


# 创建更丰富的粗粒度类别映射（基于视觉特征）
def create_species_visual_coarse_labels():
    """基于物种和视觉特征的粗粒度划分"""
    visual_groups = {
        0: "短毛猫",  # 短毛、体型中等
        1: "长毛猫",  # 长毛、体型中等或大
        2: "特殊特征猫",  # 无毛、特殊毛色等
        3: "大型犬",  # 体型大，多为工作犬
        4: "中小型犬",  # 体型中等，通用犬种
        5: "玩具犬",  # 体型小，适合家庭宠物
        6: "梗犬类",  # 梗犬特征明显
        7: "工作/牧羊犬",  # 有明显功能特征的犬种
    }

    breed_to_coarse = {
        # 猫品种 - 按毛长和特征划分
        'abyssinian': 0,  # 短毛猫
        'bengal': 0,  # 短毛猫
        'british_shorthair': 0,  # 短毛猫
        'egyptian_mau': 0,  # 短毛猫
        'russian_blue': 0,  # 短毛猫
        'siamese': 0,  # 短毛猫

        'birman': 1,  # 长毛猫
        'maine_coon': 1,  # 长毛猫
        'persian': 1,  # 长毛猫
        'ragdoll': 1,  # 长毛猫

        'bombay': 2,  # 特殊特征（黑猫）
        'sphynx': 2,  # 特殊特征（无毛）

        # 狗品种 - 按体型和功能划分
        'american_bulldog': 3,  # 大型犬
        'great_pyrenees': 3,  # 大型犬
        'leonberger': 3,  # 大型犬
        'newfoundland': 3,  # 大型犬
        'saint_bernard': 3,  # 大型犬

        'american_pit_bull_terrier': 4,  # 中小型犬
        'basset_hound': 4,  # 中小型犬
        'beagle': 4,  # 中小型犬
        'boxer': 4,  # 中小型犬
        'shiba_inu': 4,  # 中小型犬
        'staffordshire_bull_terrier': 4,  # 中小型犬

        'chihuahua': 5,  # 玩具犬
        'havanese': 5,  # 玩具犬
        'japanese_chin': 5,  # 玩具犬
        'miniature_pinscher': 5,  # 玩具犬
        'pomeranian': 5,  # 玩具犬
        'pug': 5,  # 玩具犬
        'yorkshire_terrier': 5,  # 玩具犬

        'scottish_terrier': 6,  # 梗犬
        'wheaten_terrier': 6,  # 梗犬

        'english_cocker_spaniel': 7,  # 工作/牧羊犬
        'english_setter': 7,  # 工作/牧羊犬
        'german_shorthaired': 7,  # 工作/牧羊犬
        'keeshond': 7,  # 工作/牧羊犬
        'samoyed': 7,  # 工作/牧羊犬
    }

    return visual_groups, breed_to_coarse

# 获取视觉分组和映射
visual_groups, breed_to_coarse = create_species_visual_coarse_labels()

# Oxford-IIIT Pet数据集的品种顺序（从0到36）
label_to_breed = [
    'abyssinian',  # 0
    'american_bulldog',  # 1
    'american_pit_bull_terrier',  # 2
    'basset_hound',  # 3
    'beagle',  # 4
    'bengal',  # 5
    'birman',  # 6
    'bombay',  # 7
    'boxer',  # 8
    'british_shorthair',  # 9
    'chihuahua',  # 10
    'egyptian_mau',  # 11
    'english_cocker_spaniel',  # 12
    'english_setter',  # 13
    'german_shorthaired',  # 14
    'great_pyrenees',  # 15
    'havanese',  # 16
    'japanese_chin',  # 17
    'keeshond',  # 18
    'leonberger',  # 19
    'maine_coon',  # 20
    'miniature_pinscher',  # 21
    'newfoundland',  # 22
    'persian',  # 23
    'pomeranian',  # 24
    'pug',  # 25
    'ragdoll',  # 26
    'russian_blue',  # 27
    'saint_bernard',  # 28
    'samoyed',  # 29
    'scottish_terrier',  # 30
    'shiba_inu',  # 31
    'siamese',  # 32
    'sphynx',  # 33
    'staffordshire_bull_terrier',  # 34
    'wheaten_terrier',  # 35
    'yorkshire_terrier'  # 36
]

# 创建细粒度到粗粒度的映射（基于视觉特征）
fine2coarse_id = torch.zeros(NUM_CLASSES, dtype=torch.long)
for i in range(NUM_CLASSES):
    breed = label_to_breed[i]
    fine2coarse_id[i] = breed_to_coarse[breed]

# CutMix参数
cutmix_beta = 1.0


# ==================== 自定义增强变换 ====================

class RandomGaussianBlur:
    """随机高斯模糊"""

    def __init__(self, radius_range=(0.1, 2.0), p=0.2):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomPerspective:
    """随机透视变换"""

    def __init__(self, distortion_scale=0.3, p=0.2):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return T.RandomPerspective(
                distortion_scale=self.distortion_scale,
                p=1.0
            )(img)
        return img


class RandomBrightnessContrast:
    """随机调整亮度和对比度"""

    def __init__(self, brightness=0.2, contrast=0.2, p=0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 随机调整亮度
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

            # 随机调整对比度
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        return img


class ColorJitterPlus:
    """增强版颜色抖动，包含饱和度、色调调整"""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.base_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.base_jitter(img)
        return img


class RandomRotationWithResize:
    """随机旋转并调整大小以避免黑边"""

    def __init__(self, degrees=15, p=0.2):
        self.degrees = degrees
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            # 旋转图像
            img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
            # 调整回原始大小
            img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.BICUBIC)
        return img


class MixupCutmixCollate:
    """同时支持Mixup和Cutmix的数据增强"""

    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)

        # 随机选择Mixup或Cutmix
        if random.random() < self.prob:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            bsz = images.size(0)
            index = torch.randperm(bsz)

            mixed_images = lam * images + (1 - lam) * images[index]

            return {
                "image": mixed_images,
                "label_a": labels,
                "label_b": labels[index],
                "lam": lam,
                "aug_type": "mixup"
            }
        else:
            # Cutmix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bsz = images.size(0)
            index = torch.randperm(bsz)
            W, H = images.size(-1), images.size(-2)

            cut_rat = math.sqrt(1. - lam)
            cut_w = max(1, int(W * cut_rat))
            cut_h = max(1, int(H * cut_rat))

            bbx1 = torch.randint(0, W - cut_w + 1, (1,)).item()
            bby1 = torch.randint(0, H - cut_h + 1, (1,)).item()
            bbx2 = bbx1 + cut_w
            bby2 = bby1 + cut_h

            images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)

            return {
                "image": images,
                "label_a": labels,
                "label_b": labels[index],
                "lam": lam,
                "aug_type": "cutmix"
            }


# ==================== 数据增强策略配置 ====================

def get_transform(mode='train', augment_level='medium', img_size=224):
    """
    获取数据增强变换

    Args:
        mode: 'train', 'val', 'test'
        augment_level: 'none', 'light', 'medium', 'strong', 'auto'
        img_size: 图像大小
    """
    # 基础转换
    base_transforms = [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=OXFORD_IIIT_PET_MEAN, std=OXFORD_IIIT_PET_STD)
    ]

    if mode == 'train':
        if augment_level == 'none':
            transforms_list = [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=OXFORD_IIIT_PET_MEAN, std=OXFORD_IIIT_PET_STD)
            ]

        elif augment_level == 'light':
            transforms_list = [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=OXFORD_IIIT_PET_MEAN, std=OXFORD_IIIT_PET_STD),
                RandomErasing(p=0.1, scale=(0.02, 0.1))
            ]

        elif augment_level == 'medium':
            transforms_list = [
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                ColorJitterPlus(brightness=0.15, contrast=0.15,
                                saturation=0.1, hue=0.05, p=0.5),
                RandomGaussianBlur(radius_range=(0.1, 1.0), p=0.1),
                RandomBrightnessContrast(brightness=0.15, contrast=0.15, p=0.3),
                T.ToTensor(),
                T.Normalize(mean=OXFORD_IIIT_PET_MEAN, std=OXFORD_IIIT_PET_STD),
                RandomErasing(p=0.2, scale=(0.02, 0.15), value='random')
            ]

        elif augment_level == 'strong':
            transforms_list = [
                T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.05),
                ColorJitterPlus(brightness=0.2, contrast=0.2,
                                saturation=0.15, hue=0.1, p=0.7),
                RandomGaussianBlur(radius_range=(0.1, 2.0), p=0.2),
                RandomPerspective(distortion_scale=0.2, p=0.2),
                RandomBrightnessContrast(brightness=0.2, contrast=0.2, p=0.4),
                RandomRotationWithResize(degrees=10, p=0.2),
                T.ToTensor(),
                T.Normalize(mean=OXFORD_IIIT_PET_MEAN, std=OXFORD_IIIT_PET_STD),
                RandomErasing(p=0.3, scale=(0.02, 0.2), value='random')
            ]

        elif augment_level == 'auto':
            # 使用AutoAugment策略
            transforms_list = [
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                AutoAugment(AutoAugmentPolicy.IMAGENET),
                T.ToTensor(),
                T.Normalize(mean=OXFORD_IIIT_PET_MEAN, std=OXFORD_IIIT_PET_STD),
                RandomErasing(p=0.2, scale=(0.02, 0.15))
            ]

        else:
            raise ValueError(f"Unknown augment_level: {augment_level}")

    else:  # val/test
        transforms_list = base_transforms

    return T.Compose(transforms_list)


def get_test_time_augmentation(img_size=224):
    """测试时增强（TTA）"""
    tta_transforms = []

    # 1. 原始图像
    tta_transforms.append(T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=OXFORD_IIIT_PET_MEAN, std=OXFORD_IIIT_PET_STD)
    ]))

    # 2. 水平翻转
    tta_transforms.append(T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=OXFORD_IIIT_PET_MEAN, std=OXFORD_IIIT_PET_STD)
    ]))

    # 3. 轻微颜色抖动
    tta_transforms.append(T.Compose([
        T.Resize((img_size, img_size)),
        T.ColorJitter(brightness=0.05, contrast=0.05),
        T.ToTensor(),
        T.Normalize(mean=OXFORD_IIIT_PET_MEAN, std=OXFORD_IIIT_PET_STD)
    ]))

    return tta_transforms


# ==================== 数据集包装器 ====================

class OxfordIIITPetDataset(torch.utils.data.Dataset):
    """Oxford-IIIT Pet数据集包装器"""

    def __init__(self, split="train", transform=None, tta_transforms=None):
        # 使用本地数据集，不尝试从远程下载
        self.dataset = hf_datasets.load_dataset(
            "./data/oxford-iiit-pet",
            split=split,
            download_mode=hf_datasets.DownloadMode.REUSE_DATASET_IF_EXISTS
        )
        self.transform = transform
        self.tta_transforms = tta_transforms  # 测试时增强的变换列表

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]

        if self.tta_transforms:
            # 测试时增强：返回多个增强版本
            images = [transform(image) for transform in self.tta_transforms]
            return images, label
        elif self.transform:
            image = self.transform(image)
            return image, label
        else:
            # 如果没有指定transform，只返回PIL图像
            return image, label


# ==================== 数据加载器工厂函数 ====================

def create_data_loaders(
        batch_size=32,
        img_size=224,
        augment_level='strong',
        num_workers=4,
        pin_memory=True,
        use_mixup_cutmix=True
):
    """
    创建训练、验证和测试数据加载器

    Args:
        batch_size: 批次大小
        img_size: 图像大小
        augment_level: 增强级别 ('none', 'light', 'medium', 'strong', 'auto')
        num_workers: 数据加载工作进程数
        pin_memory: 是否固定内存
        use_mixup_cutmix: 是否使用Mixup/Cutmix
    """

    # 创建数据集
    train_dataset = OxfordIIITPetDataset(
        split="train",
        transform=get_transform('train', augment_level, img_size)
    )

    val_dataset = OxfordIIITPetDataset(
        split="test",  # 注意：Oxford-IIIT Pet使用test作为验证集
        transform=get_transform('val', 'none', img_size)
    )

    # 创建collate函数
    collate_fn = None
    if use_mixup_cutmix:
        collate_fn = MixupCutmixCollate(
            mixup_alpha=1.0,
            cutmix_alpha=1.0,
            prob=0.5
        )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn if collate_fn else None,
        drop_last=True  # 丢弃最后不完整的批次，有助于稳定训练
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def create_tta_data_loader(batch_size=32, img_size=224, num_workers=4):
    """创建测试时增强的数据加载器"""
    tta_transforms = get_test_time_augmentation(img_size)

    test_dataset = OxfordIIITPetDataset(
        split="test",
        tta_transforms=tta_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


# ==================== 辅助函数 ====================

def visualize_augmentations(dataset, num_samples=5):
    """可视化数据增强效果"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_samples, 6, figsize=(15, 3 * num_samples))

    for i in range(num_samples):
        # 获取原始图像
        original_img, label = dataset.dataset[i]
        original_img = original_img.convert("RGB")

        # 应用不同级别的增强
        augment_levels = ['none', 'light', 'medium', 'strong', 'auto']

        # 显示原始图像
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Original\nLabel: {label}")
        axes[i, 0].axis('off')

        # 显示不同增强级别的结果
        for j, level in enumerate(augment_levels, 1):
            transform = get_transform('train', level)
            augmented_img = transform(original_img)

            # 反归一化以便显示
            augmented_img_np = augmented_img.numpy().transpose(1, 2, 0)
            augmented_img_np = augmented_img_np * np.array(OXFORD_IIIT_PET_STD) + np.array(OXFORD_IIIT_PET_MEAN)
            augmented_img_np = np.clip(augmented_img_np, 0, 1)

            axes[i, j].imshow(augmented_img_np)
            axes[i, j].set_title(f"Aug: {level}")
            axes[i, j].axis('off')

    plt.tight_layout()
    return fig


# 原始函数保持兼容性
def cutmix_collate(batch):
    """返回 CutMix 后的 tensor + 两个 target + lam，使用dict格式避免索引错位"""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lam = torch.distributions.Beta(cutmix_beta, cutmix_beta).sample().item()
    bsz = images.size(0)
    index = torch.randperm(bsz)
    W, H = images.size(-1), images.size(-2)

    cut_rat = math.sqrt(1. - lam)
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))

    bbx1 = torch.randint(0, W - cut_w + 1, (1,)).item()
    bby1 = torch.randint(0, H - cut_h + 1, (1,)).item()
    bbx2 = bbx1 + cut_w
    bby2 = bby1 + cut_h

    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)

    return {"image": images, "label_a": labels, "label_b": labels[index], "lam": lam}


if __name__ == "__main__":
    # 示例用法
    train_loader, val_loader = create_data_loaders(
        batch_size=16,
        img_size=224,
        augment_level='medium',
        num_workers=2,
        use_mixup_cutmix=True
    )

    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")

    # 可视化增强效果
    dataset = OxfordIIITPetDataset(split="train")
    fig = visualize_augmentations(dataset, num_samples=3)
    fig.savefig('data_augmentations.png', dpi=150, bbox_inches='tight')
    print("数据增强可视化已保存到 'data_augmentations.png'")