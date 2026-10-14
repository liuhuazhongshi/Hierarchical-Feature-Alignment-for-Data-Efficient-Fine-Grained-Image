# get_data_hier.py
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CIFAR100DatasetHier(Dataset):
    """返回 img, fine_label, coarse_label"""
    def __init__(self, data, fine_labels, coarse_labels, transform=None):
        self.data = data
        self.fine_labels = fine_labels
        self.coarse_labels = coarse_labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3,32,32).transpose(1,2,0).astype('float32') / 255.0
        if self.transform:
            img = self.transform(img)
        return img, self.fine_labels[idx], self.coarse_labels[idx]

def load_cifar100_batch(filename):
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    return d['data'], np.array(d['fine_labels']), np.array(d['coarse_labels'])

def load_cifar100_data(data_dir):
    train_data, train_fine, train_coarse = load_cifar100_batch(os.path.join(data_dir, 'train'))
    test_data, test_fine, test_coarse = load_cifar100_batch(os.path.join(data_dir, 'test'))
    return train_data, train_fine, train_coarse, test_data, test_fine, test_coarse

def get_cifar_100(batch_size=128, data_dir='./data/cifar-100-python', seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5071,0.4867,0.4408],[0.2675,0.2565,0.2761])
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071,0.4867,0.4408],[0.2675,0.2565,0.2761])
    ])

    train_data, train_fine, train_coarse, test_data, test_fine, test_coarse = load_cifar100_data(data_dir)

    # 这里直接使用标准 CIFAR-100 coarse->fine mapping
    coarse_to_fine_indices = [
        [4, 31, 55, 72, 95],
        [1, 32, 67, 73, 91],
        [54, 62, 70, 82, 92],
        [9, 10, 16, 28, 61],
        [0, 51, 53, 57, 83],
        [22, 39, 40, 86, 87],
        [5, 20, 25, 84, 94],
        [6, 7, 14, 18, 24],
        [3, 42, 43, 88, 97],
        [12, 17, 37, 68, 76],
        [23, 33, 49, 60, 71],
        [15, 19, 21, 35, 56],
        [2, 11, 26, 45, 98],
        [34, 38, 47, 64, 79],
        [29, 41, 74, 81, 99],
        [36, 63, 77, 93, 96],
        [8, 30, 44, 78, 90],
        [48, 50, 52, 59, 65],
        [27, 46, 66, 85, 89],
        [13, 39, 58, 75, 80]
    ]

    # Dataset
    train_dataset = CIFAR100DatasetHier(train_data, train_fine, train_coarse, transform=train_transform)
    test_dataset = CIFAR100DatasetHier(test_data, test_fine, test_coarse, transform=test_transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, coarse_to_fine_indices

