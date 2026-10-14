import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as transforms

from get_data import get_cifar_100


# 改进的权重初始化函数
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


# Wide ResNet 的基本块
class WideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(WideBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Wide ResNet 模型
class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=100, dropout_rate=0.3):
        super(WideResNet, self).__init__()
        self.in_channels = 16
        n = (depth - 4) // 6  # 每个 stage 的层数

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(WideBlock, 16 * widen_factor, n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(WideBlock, 32 * widen_factor, n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(WideBlock, 64 * widen_factor, n, stride=2, dropout_rate=dropout_rate)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)  # 增加最终dropout
        self.fc = nn.Linear(64 * widen_factor, num_classes)

        # 参数初始化
        self.apply(weights_init)

    def _make_layer(self, block, out_channels, n, stride, dropout_rate):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, dropout_rate))
        self.in_channels = out_channels
        for _ in range(1, n):
            layers.append(block(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# 改进的数据准备函数
def prepare_data(batch_size=64):
    # 增强的数据预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # 如果你可以修改get_cifar_100函数，建议加入这些transform
    trainloader, testloader = get_cifar_100(batch_size=batch_size)

    return trainloader, testloader


# 改进的训练函数
def train(model, trainloader, criterion, optimizer, scheduler, testloader, epochs=100):
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in trainloader:
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(trainloader)

        # 更新学习率
        scheduler.step()

        # 每个epoch都验证
        test_acc = evaluate(model, testloader)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"Best Test Accuracy: {best_acc:.2f}%")


# 模型评估函数
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return 100 * correct / total


# 主程序
def main():
    batch_size = 64
    trainloader, testloader = prepare_data(batch_size=batch_size)

    # 创建模型 - 使用更小的配置
    model = WideResNet(depth=16, widen_factor=8, num_classes=100, dropout_rate=0).cuda()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 改进的优化器和学习率调度
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=0.0005, nesterov=True)

    # 学习率调度
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[60, 120, 160],
                                               gamma=0.2)

    # 训练模型
    train(model, trainloader, criterion, optimizer, scheduler, testloader, epochs=200)

    # 加载最佳模型进行评估
    model.load_state_dict(torch.load('best_model.pth'))
    final_acc = evaluate(model, testloader)
    print(f"Final Test Accuracy: {final_acc:.2f}%")


if __name__ == '__main__':
    main()