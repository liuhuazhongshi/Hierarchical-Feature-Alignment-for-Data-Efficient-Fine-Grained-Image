# train_hier_consistency_safe.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging

from get_data import get_cifar_100  # 确保返回 coarse_to_fine_indices

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(filename='training_hier_safe.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# ----------------------------
# Dataset
# ----------------------------
class CIFAR100DatasetHier(Dataset):
    def __init__(self, data, fine_labels, coarse_labels, transform=None):
        self.data = data
        self.fine_labels = fine_labels
        self.coarse_labels = coarse_labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype('float32') / 255.0
        if self.transform:
            img = self.transform(img)
        return img, self.fine_labels[idx], self.coarse_labels[idx]

# ----------------------------
# WideResNet with two heads
# ----------------------------
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

class WideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        return F.relu(out)

class WideResNetHier(nn.Module):
    def __init__(self, depth=20, widen_factor=8, num_fine=100, num_coarse=20, dropout_rate=0.3):
        super().__init__()
        self.in_channels = 16
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(WideBlock, 16*widen_factor, n, 1, dropout_rate)
        self.layer2 = self._make_layer(WideBlock, 32*widen_factor, n, 2, dropout_rate)
        self.layer3 = self._make_layer(WideBlock, 64*widen_factor, n, 2, dropout_rate)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.feature_dim = 64 * widen_factor
        self.fc_fine = nn.Linear(self.feature_dim, num_fine)
        self.fc_coarse = nn.Linear(self.feature_dim, num_coarse)
        self.apply(weights_init)

    def _make_layer(self, block, out_channels, n, stride, dropout_rate):
        layers = [block(self.in_channels, out_channels, stride, dropout_rate)]
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
        return self.fc_fine(out), self.fc_coarse(out)

# ----------------------------
# Hierarchical consistency utils
# ----------------------------
def grouped_fine_probs_from_fine_logits(fine_logits, coarse_to_fine_indices):
    fine_probs = F.softmax(fine_logits, dim=1)
    B = fine_probs.size(0)
    device = fine_probs.device
    grouped = torch.zeros(B, len(coarse_to_fine_indices), device=device)
    for c_idx, fine_inds in enumerate(coarse_to_fine_indices):
        if len(fine_inds) == 0:
            continue
        inds = torch.tensor(fine_inds, dtype=torch.long, device=device)
        grouped[:, c_idx] = fine_probs[:, inds].sum(dim=1)
    grouped = grouped / (grouped.sum(dim=1, keepdim=True) + 1e-12)
    return grouped

# ----------------------------
# Data preparation wrapper
# ----------------------------
def prepare_data_hier(batch_size=128):
    trainloader, testloader, coarse_to_fine_indices = get_cifar_100(batch_size=batch_size)
    return trainloader, testloader, coarse_to_fine_indices

# ----------------------------
# Evaluate
# ----------------------------
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, fine_target, _ in testloader:
            imgs, fine_target = imgs.cuda(), fine_target.cuda()
            fine_out, _ = model(imgs)
            _, pred = fine_out.max(1)
            total += fine_target.size(0)
            correct += pred.eq(fine_target).sum().item()
    return 100.0 * correct / total

# ----------------------------
# Training loop
# ----------------------------
def train_hier(model, trainloader, testloader, coarse_to_fine_indices,
               epochs=200, lr=0.01, lambda_coarse=0.3, lambda_cons=0.5, weight_decay=5e-4):
    device = torch.device('cuda')
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        for imgs, fine_target, coarse_target in pbar:
            imgs, fine_target, coarse_target = imgs.cuda(), fine_target.cuda(), coarse_target.cuda()
            optimizer.zero_grad()
            fine_logits, coarse_logits = model(imgs)
            loss_fine = criterion(fine_logits, fine_target)
            loss_coarse = criterion(coarse_logits, coarse_target)
            # HCL loss
            grouped_probs = grouped_fine_probs_from_fine_logits(fine_logits, coarse_to_fine_indices)
            coarse_log_prob = F.log_softmax(coarse_logits, dim=1)
            loss_cons = F.kl_div(coarse_log_prob, grouped_probs.log(), reduction='batchmean', log_target=True)
            # Total loss
            loss = loss_fine + lambda_coarse*loss_coarse + lambda_cons*loss_cons
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            _, pred = fine_logits.max(1)
            total += fine_target.size(0)
            correct += pred.eq(fine_target).sum().item()
            pbar.set_postfix({'loss': running_loss/(total/imgs.size(0)), 'acc':100.*correct/total})

        scheduler.step()
        train_acc = 100.*correct/total
        test_acc = evaluate(model, testloader)
        log_msg = f"Epoch {epoch+1}/{epochs}, TrainAcc: {train_acc:.2f}%, TestAcc: {test_acc:.2f}%"
        logging.info(log_msg)
        print(log_msg)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model_hier_safe.pth')
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    return best_acc

# ----------------------------
# Main
# ----------------------------
def main():
    batch_size = 32
    trainloader, testloader, coarse_to_fine_indices = prepare_data_hier(batch_size=batch_size)
    model = WideResNetHier()
    best = train_hier(model, trainloader, testloader, coarse_to_fine_indices, epochs=200)

if __name__ == '__main__':
    main()
