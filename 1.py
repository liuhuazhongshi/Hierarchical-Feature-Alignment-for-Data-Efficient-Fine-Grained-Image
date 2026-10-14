import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm  # 导入 tqdm 库

from WideResNet import WideResNet
from get_data import get_cifar_100

# 1. 定义 FGSM 对抗攻击函数
def fgsm_attack(model, loss_fn, data, target, epsilon=0):
    data.requires_grad = True
    output = model(data)
    loss = loss_fn(output, target)

    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data


# 2. 模型、损失函数和优化器
model = WideResNet(depth=28, widen_factor=10, num_classes=100).cuda()

# 3. 选择设备（GPU 如果可用，否则使用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 定义 Free Adversarial Training (FreeAT) 训练循环
def train_freeAT(model, trainloader, testloader, epochs=100, epsilon=0.1, m=4):
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')):
            data, target = data.to(device), target.to(device)

            # 使用 clone() 确保 perturb_data 是叶子变量
            perturbed_data = data.clone().detach()
            perturbed_data.requires_grad = True  # 这里设置 requires_grad = True
            perturbed_data = perturbed_data.float()  # 确保是浮动类型

            # FreeAT 的关键：梯度累积 m 次
            for _ in range(m):
                output = model(perturbed_data)
                loss = loss_fn(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)  # 保留计算图

                # 检查是否成功计算了梯度
                if perturbed_data.grad is not None:
                    grad = perturbed_data.grad.data.sign()
                    perturbed_data = perturbed_data + epsilon * grad
                    perturbed_data = torch.clamp(perturbed_data, 0, 1)

                    optimizer.step()

                running_loss += loss.item()

        # 每个 epoch 后测试模型在测试集上的表现
        test(model, testloader, device)


# 5. 测试模型性能
def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(testloader, desc='Testing', unit='batch'):
            data, target = data.to(device), target.to(device)

            # 正常样本预测
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

    # 打印结果
    print(f'Accuracy on the test set: {100 * accuracy:.2f}%')
    print(f'Precision (Weighted): {precision:.4f}')
    print(f'Recall (Weighted): {recall:.4f}')
    print(f'F1 Score (Weighted): {f1:.4f}')

# 获取 CIFAR-100 数据
trainloader, testloader = get_cifar_100(batch_size=16)

# 6. 训练模型
train_freeAT(model, trainloader, testloader, epochs=100, epsilon=0.1, m=2)
