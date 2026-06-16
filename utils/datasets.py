# -*- coding: utf-8 -*-
"""
Module: utils/datasets.py
Decoupled from: v3_2passKNN_refactored.py
Decoupling baseline: commit 6d3088f (refactor/true-decoupled-minimal)
Description: 数据集类定义、数据增强变换、采样器
"""
import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd

from data.dataset import CUB200Partial, CIFAR10Partial, CIFAR100Partial
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy
from data.crowdsource import *


def get_pals_transforms(dataset_name):
    # --- (新增) 众包数据集的 Mean/Std ---
    if dataset_name == 'Treeversity':
        mean = [0.4439581940620345, 0.4509297096690951, 0.3691211738638277]
        std = [0.23407518616927706, 0.22764417468550843, 0.2600833107790479]
    elif dataset_name == 'Benthic':
        mean = [0.34728872821176615, 0.40013687864974884, 0.4110478166769647]
        std = [0.1286915489786319, 0.13644626747739305, 0.14258506692263767]
    elif dataset_name == 'Plankton':
        mean = [0.9663359216202008, 0.9663359216202008, 0.9663359216202008]
        std = [0.10069729102981237, 0.10069729102981237, 0.10069729102981237]
    elif dataset_name == 'CUB200':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else: # 默认为 CIFAR
        mean, std = ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]) if '100' in dataset_name else ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    # --- (修改) 扩展 Transform 逻辑 ---
    
    # (新增) CUB200 (使用 PALS 原始的强 Aug)
    if dataset_name == 'CUB200':
        weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            # CIFAR10Policy(), 
            ImageNetPolicy(),
            transforms.ToTensor(), 
            Cutout(n_holes=1, length=56), # <-- 关键！
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        
    elif dataset_name == 'Treeversity':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(int(224/0.875)), # (256)
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset_name == 'Benthic':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((112,112)),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset_name == 'Plankton':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else: # CIFAR
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    # 包含您截图中的所有新数据集
    if dataset_name in ['Turkey', 'Pig', 'MiceBone', 'QualityMRI', 'Synthetic', 
                        'verse_blended-vps', 'verse_mask1-vps', 'CIFAR10H']:
        
        # 使用 ImageNet 统计数据作为通用初始化
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # 如果是 CIFAR10H,可能图片很小 (32x32),需要特殊处理
        if 'CIFAR' in dataset_name or 'Synthetic' in dataset_name:
            resize_size = 32
            crop_size = 32
        else:
            # 其他真实世界数据集 (Turkey, Pig等) 使用标准 224
            resize_size = 256
            crop_size = 224

        weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(), # 强增强
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return weak_transform, strong_transform, test_transform

class FeatureExtractionDataset(Dataset):
    def __init__(self, base_dataset, transform): 
        self.base_dataset, self.transform = base_dataset, transform
        
        # --- (修改) ---
        # 我们需要明确区分 CUB200 和 Crowdsource
        self.is_cub = isinstance(self.base_dataset, CUB200Partial)
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
        # --- (修改结束) ---

    def __len__(self): 
        return len(self.base_dataset)
        
    def __getitem__(self, index):
        # 1. 获取原始图像
        
        # --- (修改) ---
        if self.is_cub:
            # CUB200: .data 是 DataFrame. 必须用 .data_paths
            # (这是在破坏"封装",但这是在你设定的约束下唯一可行的方法)
            img_path = os.path.join(self.base_dataset.root, 
                                    self.base_dataset.base_folder, 
                                    'images', 
                                    self.base_dataset.data_paths[index])
            img = Image.open(img_path).convert('RGB')
        
        elif self.is_crowd:
            # Crowdsource: .data 是 'list' of paths, 可以直接用 [index]
            img_path = self.base_dataset.data[index]
            img = Image.open(img_path).convert('RGB')


        else:
            # CIFAR: .data 是 numpy 数组
            img = Image.fromarray(self.base_dataset.data[index])
        # --- (修改结束) ---

        # 2. 应用 *这个类* 自己的 transform (即 test_t)
        return self.transform(img), index


# ==============================================================================
#      Section 2.5: 统一数据集 + 动态重要性采样器 (Unified Dataset + Importance Sampler)
# ==============================================================================


class ImageOnlyDataset(Dataset):
    def __init__(self, base_dataset, weak_t, strong_t):
        self.base_dataset = base_dataset
        self.weak_t = weak_t
        self.strong_t = strong_t
        self.is_cub = isinstance(self.base_dataset, CUB200Partial)
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        if self.is_cub:
            img_path = os.path.join(self.base_dataset.root, self.base_dataset.base_folder, 'images', self.base_dataset.data_paths[idx])
            img = Image.open(img_path).convert('RGB')
        elif self.is_crowd:
            img_path = self.base_dataset.data[idx]
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.fromarray(self.base_dataset.data[idx])
        return self.weak_t(img), self.strong_t(img), idx


class DynamicWeightedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_samples):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = True
    def update_weights(self, new_weights):
        self.weights = torch.as_tensor(new_weights, dtype=torch.double)
    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement)
        return iter(rand_tensor.tolist())
    def __len__(self):
        return self.num_samples


class UnifiedSSLDataset(Dataset):
    """
    统一的单流数据集，合并可靠集和不可靠集
    每个样本带有 is_reliable 标志用于区分训练策略
    """
    def __init__(self, base_dataset, data_list, weak_t, strong_t):
        """
        Args:
            base_dataset: 原始数据集 (CIFAR/CUB/Crowdsource)
            data_list: [(idx, label, is_reliable), ...]
                - idx: 原始索引
                - label: 伪标签（可靠集）或 -1（不可靠集）
                - is_reliable: True/False
            weak_t: 弱增强变换
            strong_t: 强增强变换
        """
        self.base_dataset = base_dataset
        self.data_list = data_list
        self.weak_t = weak_t
        self.strong_t = strong_t
        
        # 检测数据集类型
        self.is_cub = isinstance(self.base_dataset, CUB200Partial)
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        original_idx, label, is_reliable = self.data_list[idx]
        
        # 获取原始图像
        if self.is_cub:
            img_path = os.path.join(self.base_dataset.root,
                                    self.base_dataset.base_folder,
                                    'images',
                                    self.base_dataset.data_paths[original_idx])
            img = Image.open(img_path).convert('RGB')
        elif self.is_crowd:
            img_path = self.base_dataset.data[original_idx]
            img = Image.open(img_path).convert('RGB')
        else:  # CIFAR
            img = Image.fromarray(self.base_dataset.data[original_idx])
        
        return (self.weak_t(img), self.strong_t(img),
                label, is_reliable, original_idx)


