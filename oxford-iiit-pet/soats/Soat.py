import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0
from torchvision.transforms import RandomErasing, AutoAugmentPolicy
from torchvision.models.vision_transformer import VisionTransformer
from tqdm import tqdm
import timm
# ======================
# 配置
# ======================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
num_epochs = 300
train_ratios = [0.5,1]
log_file = "baseline_all.log"
use_cutmix = True
cutmix_beta = 1.0
seed = 42

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

torch.manual_seed(seed)
np.random.seed(seed)

# ======================
# 日志
# ======================

# Save the model checkpoint after each epoch
def save_model(model, model_name, epoch, ratio, path='model_checkpoints'):
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, f'{model_name}_epoch{epoch+1}_ratio{int(ratio*100)}.pth')
    torch.save(model.state_dict(), model_path)
    log(f"Model saved at {model_path}")

def log(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# ======================
# CutMix
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
# 数据增强
# ======================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    RandomErasing(p=0.5, scale=(0.02, 0.15))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])

# ======================
# 模型构建
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
            image_size=32,
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

        model = timm.create_model('vit_large_patch16_224', pretrained=False, img_size=32)

        # 强制修改 patch size

        model.patch_embed.patch_size = 4  # 可以修改为更小的 patch size

        # 修改最后的分类层

        model.head = nn.Linear(model.head.in_features, num_classes)

    else:
        raise ValueError("Unknown model")

    return model.to(device)

# ======================
# 训练
# ======================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    correct, total = 0, 0

    for batch in tqdm(loader, leave=False):

        if use_cutmix:
            images, y1, y2, lam = batch
            images = images.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            # 调试输出
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, y1) + (1 - lam) * criterion(outputs, y2)

        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)

        if use_cutmix:
            correct += lam * preds.eq(y1).sum().item() + \
                       (1 - lam) * preds.eq(y2).sum().item()
        else:
            correct += preds.eq(labels).sum().item()

        total += images.size(0)

    return 100 * correct / total

# ======================
# 测试
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

    if os.path.exists(log_file):
        os.remove(log_file)

    train_dataset = datasets.CIFAR100('./data',
                                      train=True,
                                      download=True,
                                      transform=transform_train)

    test_dataset = datasets.CIFAR100('./data',
                                     train=False,
                                     download=True,
                                     transform=transform_test)

    num_classes = len(train_dataset.classes)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2)

    criterion = nn.CrossEntropyLoss()

    for model_name in ['PVTv3', 'ResNet50', 'DenseNet121',  'ViT-B', 'Swin-T', 'DeiT3', 'ConvNeXt-V2', 'EfficientNet-B0',]:

        log(f"\n============================")
        log(f"Model: {model_name}")
        log(f"============================")

        for ratio in train_ratios:

            log(f"\nTrain Ratio: {int(ratio * 100)}%")

            model = build_model(model_name, num_classes)

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

            subset_len = int(len(train_dataset) * ratio)
            indices = np.random.choice(len(train_dataset),
                                       subset_len,
                                       replace=False)

            train_subset = Subset(train_dataset, indices)

            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=cutmix_collate if use_cutmix else None
            )

            for epoch in range(num_epochs):

                train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer
                )

                scheduler.step()

                test_acc = evaluate(model, test_loader)

                log(f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Test Acc: {test_acc:.2f}% | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}")
            save_model(model, model_name, epoch, ratio)
            log(f"Final Test Acc (Ratio {int(ratio*100)}%): "
                f"{test_acc:.2f}%")