import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0
from torchvision.transforms import RandomErasing, AutoAugment, AutoAugmentPolicy
from torchvision.models.vision_transformer import VisionTransformer
from tqdm import tqdm
import timm
from torchvision.datasets import ImageFolder
import torchvision

# ======================
# 配置
# ======================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32  # Stanford Cars图像更大，减小batch size
num_epochs = 300  # 可以适当减少epoch
train_ratios = [0.5, 1]
log_file = "stanford_cars_baseline.log"
use_cutmix = True
cutmix_beta = 1.0
seed = 42
img_size = 128  # 标准ImageNet尺寸

# Stanford Cars的统计数据（从ImageNet继承）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

torch.manual_seed(seed)
np.random.seed(seed)


# ======================
# 日志
# ======================
def save_model(model, model_name, epoch, ratio, path='car_checkpoints'):
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, f'{model_name}_epoch{epoch + 1}_ratio{int(ratio * 100)}.pth')
    torch.save(model.state_dict(), model_path)
    log(f"Model saved at {model_path}")


def log(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')


# ======================
# CutMix (适配224x224)
# ======================
def cutmix_collate(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])

    lam = np.random.beta(cutmix_beta, cutmix_beta)
    bsz = images.size(0)
    index = torch.randperm(bsz)

    W = images.size(3)
    H = images.size(2)

    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    return images, labels, labels[index], lam


# ======================
# 数据增强 (针对Stanford Cars优化)
# ======================
transform_train = transforms.Compose([
    transforms.Resize(256),  # 先resize到稍大尺寸
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),  # ImageNet策略
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    RandomErasing(p=0.25, scale=(0.02, 0.1)),  # 稍微减小擦除范围
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


# ======================
# 下载Stanford Cars数据集
# ======================
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
        # 如果torchvision版本不支持，提供手动下载说明
        log("Please install torchvision>=0.13.0 or download manually:")
        log("1. Download from: https://ai.stanford.edu/~jkrause/cars/car_dataset.html")
        log("2. Extract to ./data/stanford_cars/")

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


# ======================
# 模型构建 (针对224x224优化)
# ======================
def build_model(model_name, num_classes):

    if model_name == 'ResNet50':
        model = models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'ViT-B':
        model = VisionTransformer(
            image_size=128,
            patch_size=4,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=num_classes,
            dropout=0.1,
            attention_dropout=0.1
        )

    elif model_name == 'Swin-T':
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)


    elif model_name == 'DenseNet121':

        model = models.densenet121(weights=None)

        # CIFAR适配（关键）

        model.features.conv0 = nn.Conv2d(

            3, 64, kernel_size=3, stride=1, padding=1, bias=False

        )

        model.features.pool0 = nn.Identity()

        model.classifier = nn.Linear(

            model.classifier.in_features, num_classes

        )


    elif model_name == 'EfficientNet-B0':

        model = efficientnet_b0(weights=None)

        # 改stem（去stride=2）

        model.features[0][0] = nn.Conv2d(

            3, 32, kernel_size=3, stride=1, padding=1, bias=False

        )

        model.classifier[1] = nn.Linear(

            model.classifier[1].in_features, num_classes

        )

    elif model_name == 'SwinV2':
        model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=num_classes
        )

    elif model_name == 'ConvNeXt-V2':
        model = timm.create_model('convnextv2_tiny', pretrained=False)
        model.head = nn.Linear(model.head.in_features, num_classes)
        # 修改模型输入层，适应 32x32 输入
        model.stem[0] = nn.Conv2d(
            3, 96, kernel_size=4, stride=2, padding=1, bias=False
        )

        # 自定义 forward 函数
        def forward_fn(x):
            x = model.forward_features(x)  # 获取特征输出
            x = x.flatten(1)  # Flatten to shape [batch_size, num_channels]
            return x

        model.forward = forward_fn  # 重新定义 forward 方法

    elif model_name == 'PVTv3':

        model = timm.create_model('pvt_v2_b0', pretrained=False, in_chans=3, num_classes=num_classes)
        model.head = nn.Linear(model.head.in_features, num_classes)


    elif model_name == 'DeiT3':

        # CIFAR100 小图像用 patch_size=4

        model = timm.create_model('vit_large_patch16_224', pretrained=False, img_size=64)

        # 强制修改 patch size

        model.patch_embed.patch_size = 4  # 可以修改为更小的 patch size

        # 修改最后的分类层

        model.head = nn.Linear(model.head.in_features, num_classes)

    else:
        raise ValueError("Unknown model")

    return model.to(device)
# ======================
# 训练 (保持不变)
# ======================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    correct, total = 0, 0
    running_loss = 0.0

    for batch in tqdm(loader, leave=False):
        if use_cutmix and len(batch) == 4:  # CutMix返回4个值
            images, y1, y2, lam = batch
            images = images.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, y1) + (1 - lam) * criterion(outputs, y2)

            _, preds = outputs.max(1)
            correct += lam * preds.eq(y1).sum().item() + \
                       (1 - lam) * preds.eq(y2).sum().item()
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += images.size(0)

    return 100 * correct / total, running_loss / len(loader)


# ======================
# 测试 (保持不变)
# ======================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = outputs.max(1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# ======================
# 主程序
# ======================
if __name__ == '__main__':
    # 清空日志
    if os.path.exists(log_file):
        os.remove(log_file)

    log("=" * 50)
    log("Stanford Cars Baseline Experiment")
    log(f"Device: {device}")
    log(f"Batch Size: {batch_size}")
    log(f"Image Size: {img_size}")
    log(f"CutMix: {use_cutmix}")
    log("=" * 50)

    # 加载数据集
    log("\nLoading Stanford Cars dataset...")
    train_dataset, test_dataset = get_stanford_cars('./data')
    num_classes = len(train_dataset.classes)
    log(f"Number of classes: {num_classes}")
    log(f"Training samples: {len(train_dataset)}")
    log(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    # 'EfficientNet-B0','Swin-T',, 'ViT-B','PVTv3'
    # 定义要测试的模型（可以精简一些） 'ConvNeXt-V2', 'DeiT3',
    model_list = ['ResNet50', 'DenseNet121']

    for model_name in model_list:
        log(f"\n{'=' * 40}")
        log(f"Model: {model_name}")
        log(f"{'=' * 40}")

        for ratio in train_ratios:
            log(f"\n>>> Train Ratio: {int(ratio * 100)}%")

            # 构建模型（使用预训练权重）
            model = build_model(model_name, num_classes)

            # 计算可训练参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f"Total parameters: {total_params / 1e6:.2f}M")
            log(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

            # 优化器（分层学习率）
            # 统一使用 AdamW（更公平）
            optimizer = optim.AdamW(
                model.parameters(),
                lr=0.001,
                weight_decay=5e-4
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs
            )

            # 创建训练子集
            subset_len = int(len(train_dataset) * ratio)
            indices = np.random.choice(len(train_dataset),
                                       subset_len,
                                       replace=False)
            train_subset = Subset(train_dataset, indices)

            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=cutmix_collate if use_cutmix else None
            )

            best_test_acc = 0
            patience_counter = 0
            patience = 20  # 早停

            for epoch in range(num_epochs):
                # 训练
                train_acc, train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer
                )
                print(f'训练次数：{epoch},训练级准确率：{train_acc}, 训练loss：{train_loss}')
                # 调度器更新
                scheduler.step()
                # if epoch>100:
                    # 测试
                test_acc = evaluate(model, test_loader)

                # 保存最佳模型
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                    save_model(model, model_name, epoch, ratio)
                else:
                    patience_counter += 1

                # 日志
                log(f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Test Acc: {test_acc:.2f}% | "
                    f"Best: {best_test_acc:.2f}% | "
                    f"LR: {optimizer.param_groups[-1]['lr']:.6f}")


                log(f"\nFinal Results for {model_name} (Ratio {int(ratio * 100)}%):")
                log(f"Best Test Accuracy: {best_test_acc:.2f}%")