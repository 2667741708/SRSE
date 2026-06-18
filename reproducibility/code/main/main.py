# -*- coding: utf-8 -*-
"""SRSE main training entry point.

Reliable samples are selected by the SRSE class-balanced policy. The selector
first keeps predictions whose argmax label is supported by the source prior,
then applies a per-predicted-class quota controlled by ``delta`` and ranks
samples by classwise discrepancy. Legacy reliable-set mode switching is not
part of the public SRSE interface.

Second-pass fusion applies the source prior only to the model branch:

    prior_effective = normalize(omega * p_model_effective)
    propagation_input_2 = normalize(r_i * p_knn1 + (1 - r_i) * prior_effective)

This keeps the graph branch able to restore source-missing labels through
neighborhood evidence while constraining model evidence by the source prior.
Experiment commands are maintained under ``reproducibility/commands``.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math # <--- Added
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import argparse
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time
from collections import deque
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import logging
import csv
import json
import shutil
from PIL import Image
try:
    from torch.amp import autocast as _autocast, GradScaler
    def autocast():
        return _autocast(device_type='cuda')
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import wandb
import sys
import itertools
import datetime # <--- Added
from torchvision.models import resnet18, resnet50
# 假设您的工具函数在以下路径
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy ,ImageNetPolicy
# from utils.prototype_manager import PrototypeManager
# from utils.diagnostics import log_tri_consensus_diagnostics
from data.crowdsource import *
from data.dataset import CIFAR10Partial, CIFAR100Partial
# 5. 创建采样器 (直接使用对齐后的列表)
# 注意:sampling_weights_aligned 的长度必须等于 unified_dataset 的长度
from torch.utils.data import WeightedRandomSampler
# ==============================================================================
#                      Section 0: 环境设置 (Environment Setup)
# ==============================================================================
def set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# In Section 0: 环境设置 (Environment Setup)

def setup_logger(log_dir, filename="run.log", is_master=False, to_console=False, append=False):
    """
    Modified to allow disabling console output explicitly.
    """
    logger_name = f"logger_{log_dir.replace('/', '_')}_{filename}"
    logger = logging.getLogger(logger_name)
    
    if logger.hasHandlers():
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
            
    logger.setLevel(logging.INFO)
    logger.propagate = False 
    
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S")
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)
    
    # File Handler (Always active)
    file_handler = logging.FileHandler(log_file, mode='a' if append else 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler (Only active if to_console is True)
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

def _resolve_resume_checkpoint(args, log_dir):
    resume_checkpoint = getattr(args, 'resume_checkpoint', '')
    if resume_checkpoint:
        return resume_checkpoint
    if getattr(args, 'auto_resume', False):
        latest_path = os.path.join(log_dir, 'latest.pt')
        if os.path.exists(latest_path):
            return latest_path
    return None

def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _rng_state_dict():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state

def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])

def _prototype_state_dict(proto_manager):
    return {
        name: value.detach().cpu() if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in vars(proto_manager).items()
        if not name.startswith('__')
    }

def _load_prototype_state_dict(proto_manager, state, device):
    if not state:
        return
    for name, value in state.items():
        if torch.is_tensor(value):
            setattr(proto_manager, name, value.to(device))
        else:
            setattr(proto_manager, name, value)

def _temporal_state_dict(state_manager):
    return {
        name: value.detach().cpu() if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in vars(state_manager).items()
        if not name.startswith('__')
    }

def _load_temporal_state_dict(state_manager, state):
    if not state:
        return
    for name, value in state.items():
        setattr(state_manager, name, value)

def _append_epoch_metrics(log_dir, metrics):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'epoch_metrics.csv')
    jsonl_path = os.path.join(log_dir, 'epoch_metrics.jsonl')
    fieldnames = list(metrics.keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    with open(jsonl_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False) + '\n')

def _save_epoch_checkpoint(log_dir, epoch, args, encoder, classifier, optimizer, scheduler,
                           proto_manager, state_manager, best_test_acc, test_acc, metrics, is_best=False):
    os.makedirs(log_dir, exist_ok=True)
    checkpoint = {
        'epoch': int(epoch),
        'next_epoch': int(epoch),
        'args': vars(args),
        'encoder': encoder.state_dict(),
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'prototype_manager': _prototype_state_dict(proto_manager),
        'temporal_state_manager': _temporal_state_dict(state_manager),
        'best_test_acc': float(best_test_acc),
        'test_acc': float(test_acc),
        'metrics': metrics,
        'rng_state': _rng_state_dict(),
    }
    epoch_path = os.path.join(log_dir, f'epoch_{epoch:04d}.pt')
    latest_path = os.path.join(log_dir, 'latest.pt')
    torch.save(checkpoint, epoch_path)
    shutil.copyfile(epoch_path, latest_path)
    if is_best:
        shutil.copyfile(epoch_path, os.path.join(log_dir, 'best.pt'))
    return epoch_path

def apply_external_source_if_requested(args, dataset, logger, log_dir):
    source_path = getattr(args, 'external_source_path', '')
    if not source_path:
        return False
    path = os.path.abspath(source_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"External source prior not found: {path}")
    if path.endswith('.npz'):
        archive = np.load(path)
        key = 'source_prior' if 'source_prior' in archive else archive.files[0]
        source_prior = archive[key]
    elif path.endswith('.npy'):
        source_prior = np.load(path)
    else:
        source_prior = np.loadtxt(path, delimiter=',')
    source_prior = np.asarray(source_prior, dtype=np.float32)
    if not hasattr(dataset, 'soft_labels'):
        raise AttributeError('Dataset does not expose soft_labels for external source replacement.')
    if source_prior.shape != np.asarray(dataset.soft_labels).shape:
        raise ValueError(f"External source shape {source_prior.shape} does not match soft_labels {np.asarray(dataset.soft_labels).shape}.")
    dataset.soft_labels = source_prior.copy()
    dataset.original_soft_labels = source_prior.copy()
    os.makedirs(log_dir, exist_ok=True)
    np.save(os.path.join(log_dir, 'external_source_prior.npy'), source_prior)
    logger.info(f"Loaded external source prior from {path}")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='SRSE training entry with class-balanced reliable selection')
    # 基本设置
    parser.add_argument('--exp_name', type=str, default='SRSE_Run', help='Experiment name.')
    
    # 在 parse_args() 函数中修改:
    parser.add_argument('--dataset', type=str, default='CIFAR100', 
                        choices=['CIFAR10', 'CIFAR100', 'CIFAR100H', 
                                'Treeversity', 'Benthic', 'Plankton',])
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--out', type=str, default='./out_ultimate', help='Directory for output')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1], help='List of random seeds.')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    parser.add_argument('--external_source_path', type=str, default='',
                        help='Optional .npy/.npz/.csv source prior replacing generated candidate labels.')
    
    # 部分标签 (PLL) 设置
    parser.add_argument('--pr', type=float, default=0.05, help='partial ratio (q)')
    parser.add_argument('--nr', type=float, default=0.5, help='noise ratio (eta)')
    parser.add_argument('--lpi', type=int, default=10, help='Labels Per Image (LPI) for crowdsource NPLL conversion')
    parser.add_argument('--slice', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Fold slice index for cross-validation')
    parser.add_argument('--split_protocol', type=str, default='standard', choices=['standard', 'pals_3fold'],
                        help='Crowdsourced-dataset fold protocol.')
    # 训练超参数
    parser.add_argument('--network', type=str, default='R18', help='Network architecture (R18, R50)')
    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs.')
    parser.add_argument('--max_run_epochs', type=int, default=0,
                        help='Optional execution cap for smoke or epoch-level validation. '
                             'Keeps --epochs as the schedule horizon; 0 runs all epochs.')
    parser.add_argument('--resume_checkpoint', type=str, default='',
                        help='Checkpoint path to resume from.')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Resume from latest.pt in the run directory when it exists.')
    parser.add_argument('--checkpoint_every_epoch', action='store_true',
                        help='Save epoch_XXXX.pt, latest.pt, best.pt, epoch_metrics.csv, and epoch_metrics.jsonl.')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step'],
                        help='Type of learning rate scheduler (cosine or step).')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[60, 120, 160, 200],
                        help='Epoch milestones for the step learning rate scheduler.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='Decay rate (gamma) for the step learning rate scheduler.')    
    # 损失函数超参数
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Alpha for Mixup.')
    parser.add_argument('--lsr', type=float, default=0.5, help='Label smoothing rate.')
    parser.add_argument('--consistency_weight', type=float, default=1.0, help='Weight for consistency loss.')
    parser.add_argument('--feature_extract_view', type=str, default='weak_strong_fusion',
                        choices=['weak_only', 'weak_strong_fusion'],
                        help='Image view used for feature extraction.')

    # --- 🚀 消融实验开关 (Ablation Study Flags) ---
    parser.add_argument('--no_reliable_mixup', action='store_true', help='[Ablation] Disable Mixup on reliable set.')
    parser.add_argument('--ablate_no_cr', action='store_true', help='[Ablation] Disable consistency regularization on the strong view.')
    parser.add_argument('--no_rebalance', action='store_true', help='[Ablation] Disable class rebalancing on pseudo-labels.')
    parser.add_argument('--no_softmatch', action='store_true', help='[Ablation] Disable SoftMatch weighting (force weight=1.0).')
    parser.add_argument('--no_unreliable_mixup', action='store_true', help='[Ablation] Disable Mixup on unreliable set (use standard consistency).')
    
    # [新增] 完全禁用不可靠集训练
    parser.add_argument('--no_unreliable_training', action='store_true', help='[Ablation] COMPLETELY ignore unreliable set (Supervised Only).')
    
    # [新增] 消融:禁用 Middleware Rectification
    parser.add_argument('--no_rectify', action='store_true', help='[Ablation] Disable Middleware Gating/Rectification.')
    
    # [新增] 消融:EMA 因子
    parser.add_argument('--ema_alpha', type=float, default=0.999, help='EMA momentum factor (default: 0.999).')
    # ---------------------------------------------

    # KNN & 平衡参数
    parser.add_argument('--k_val', type=int, default=15, help='k for knn')
    parser.add_argument('--delta', type=float, default=0.25, help='example selection quantile')
    
    parser.add_argument('--history_len', type=int, default=15, help='example selection quantile')
    # --- 🚀 [Added for Ablation Master Control] ---
    parser.add_argument('--consensus_power', type=float, default=2.0, help='Power for consensus proportion in dynamic weight (default: 2.0)')
    parser.add_argument('--fix_dynamic_weight', action='store_true', help='[Ablation] Fix dynamic consistency weight to 1.0 (Disable curriculum)')
    # ----------------------------------------------
# --- 修改部分开始 ---
    parser.add_argument('--sim_mode_1', type=str, default='topology',  # <--- 修改默认值为 topology
                        choices=['topology', 'exp', 'daes', 'none','topology_daes'],   # <--- 确保包含所有选项
                        help='Similarity measure for Stage 1')

    parser.add_argument('--sim_mode_2', type=str, default='daes',      # <--- 第二阶段通常用 daes 或 topology
                        choices=['topology', 'exp', 'daes', 'none', 'topology_daes'],   # <--- 确保包含所有选项
                        help='Similarity measure for Stage 2')   

    # [Topology] 信誉度核:让 topology 在 one-hot 标签下也能工作
    parser.add_argument('--topology_rel_mode', type=str, default='masked_entropy',
                        choices=['masked_entropy', 'kl', 'agree'],
                        help='[Topology] Reliability score mode. Use kl/agree to support one-hot labels.')
    parser.add_argument('--topology_rel_gamma', type=float, default=2.0,
                        help='[Topology] Penalty strength for low-consensus nodes (default: 2.0).')
    parser.add_argument('--topology_rel_eps', type=float, default=1e-12,
                        help='[Topology] Epsilon for numerical stability.')

    parser.add_argument('--warmup_epochs', type=int, default=250, help='Epochs for linear LR warmup.')

    # 日志
    parser.add_argument('--detailed_log', action='store_true', help='Enable detailed diagnostic logging.')
    # 1. Teacher Gating 控制 (干预机制)
    parser.add_argument('--gating_start_ratio', type=float, default=0.2, 
                        help='[Gating] Ratio of epochs before teacher gating starts (default: 0.2, means start at 20%% epoch).')
    parser.add_argument('--gating_max_alpha', type=float, default=1.0, 
                        help='[Gating] Max influence of teacher (0.0 to 1.0). Set <1.0 to always keep some geometry signal.')

    # 2. DAES 算法控制 (拓扑构建)
    parser.add_argument('--daes_clamp', type=float, default=0.25, 
                        help='[DAES] Max temperature clamp (Anti-oversmoothing lock). Lower is sharper.')
    parser.add_argument('--daes_entropy_weight', type=float, default=0.2, 
                        help='[DAES] Sensitivity to local entropy (tau = base + weight * H).')
    parser.add_argument('--daes_sharpening_power', type=float, default=2.0, 
                        help='[DAES] Sharpening power for local mean calculation (default=1.0).')
    
    # [新增] DAES 参数化控制
    parser.add_argument('--daes_spatial_temp', type=float, default=0.5, 
                        help='[DAES] Temperature for spatial weighting (default: 0.5).')
    parser.add_argument('--daes_base_tau', type=float, default=0.1, 
                        help='[DAES] Base temperature for affinity matrix (default: 0.1).')
    parser.add_argument('--daes_entropy_coeff', type=float, default=0.5, 
                        help='[DAES] Coefficient for entropy-based temperature adjustment (default: 0.5).')
    parser.add_argument('--daes_sim_power', type=float, default=2.0, 
                        help='[DAES] Power to raise similarity to (default: 2.0).')
    
    # [新增] 拓扑参考模式
    parser.add_argument('--daes_topology_ref_mode', type=str, default='hard', choices=['gated', 'hard'],
                        help='[DAES] Topology reference mode: "gated" (use teacher confidence gated signal) or "hard" (use raw hard signal).')

    # 3. KNN 图构建
    parser.add_argument('--knn_heads', type=int, default=4, 
                        help='[KNN] Number of  heads for metric learning (Robustness).')
    
    # 🚀 新增: 双源KNN交集筛选参数
    
    # 🚀 新增: 动态全可靠集训练参数
    # 在 parse_args() 函数的 "核心算法开关" 或 "消融实验" 部分加入:


    # [新增] 消融:禁用一致性正则化
    parser.add_argument('--enable_knn1_model_fuse', action='store_true',
                        help='[EXP9] Enable model-geometry fusion with KNN1 scores before candidate projection.')
    
    parser.add_argument('--fusion_mode', type=str, default='weighted_sum', 
                        choices=['geometric', 'weighted_sum'],
                        help='Fusion mode for model prediction and KNN scores (default: weighted_sum)')

    # ===========================================================================
    # [消融保留] kl_self_mode（结构保留，本脚本主要使用自适应融合流程）
    # ===========================================================================
    parser.add_argument('--kl_self_mode', type=str, default='with_self',
                        choices=['with_self', 'no_self'],
                        help='[Ablation/KL] Self-node inclusion in KL reliability score.')

    # ===========================================================================
    # [新增 / New] 自适应可靠性融合参数
    # 修改来源: 结合投影_众包_自节点_模型预测融合_kl消融.py
    #           -> 结合投影_众包_自节点_自适应可靠融合.py
    # 修改位置: parse_args() 末尾
    # 修改内容: 新增三个参数控制自适应CE可靠性融合行为
    # Modified: Added three params to control adaptive CE-based reliability fusion.
    # ===========================================================================
    parser.add_argument('--adap_rel_gamma', type=float, default=2.0,
                        help='[AdapFuse] Decay factor γ for reliability score r_i=exp(-γ*(CE/logC)²). '
                             'Higher γ makes the gate more aggressive.')
    parser.add_argument('--adap_rel_eps', type=float, default=1e-12,
                        help='[AdapFuse] Numerical stability ε for CE computation.')
    parser.add_argument('--adap_ce_direction', type=str, default='knn_over_model',
                        choices=['knn_over_model', 'model_over_knn'],
                        help='[AdapFuse] CE direction for reliability scoring.\n'
                             '  knn_over_model (default): ce=-∑ p_knn2*log(p_model). '
                             'Low r_i when model is random (early epochs) → prior dominates → '
                             'candidate-set true-label info guides KNN (natural curriculum).\n'
                             '  model_over_knn: ce=-∑ p_model*log(p_knn2). '
                             'Early: measures KNN entropy, rewards already-sharp KNN distributions.')

    # ===========================================================================
    # [新增 / New] 渐进融合(Progressive Fusion)预热参数
    # 修改来源: 结合投影_众包_自节点_自适应可靠融合.py
    #           -> 结合投影_众包_自节点_自适应渐进融合.py
    # 修改内容: 新增 model_warmup_epochs 参数，控制模型预测权重的预热轮数
    # Modified: Added model_warmup_epochs to control w_model warmup schedule.
    # ===========================================================================
    parser.add_argument('--model_warmup_epochs', type=int, default=10,
                        help='[ProgFuse] Number of warmup epochs for model prediction weight. '
                             'w_model = max_w_model * min(1.0, epoch / model_warmup_epochs). '
                             'At epoch 0, w_model=0 -> p_model_effective = p_knn1 (pure first-pass KNN). '
                             'At epoch >= model_warmup_epochs, w_model=max_w_model -> capped model/KNN mixture.')
    # [恢复] max_w_model 参数 / Restored max_w_model parameter
    parser.add_argument('--max_w_model', type=float, default=0.5,
                        help='Maximum value for w_model (default: 0.5).')

    # ===========================================================================
    # [新增 / New] 自适应连续传播深度控制参数
    # [MODIFIED - gitdiffer]
    # <<<<<<< BASE: 
    # =======
    parser.add_argument("--adaptive_prop_depth", action="store_true",
                        help="[AdapDepth] Automatically truncate propagation (skip Stage 3) if over-sharpening is detected (high consensus but low reliable volume).")
    parser.add_argument("--expected_rel_ratio", type=float, default=0.5,
                        help="[AdapDepth] Expected minimum ratio of reliable samples (default: 0.5). Used with --adaptive_prop_depth.")
    # >>>>>>> NEW: 根据可靠集数量和共识率动态截断传播深度。
    # ===========================================================================

    return parser.parse_args()
# (在 Section 2: 数据处理与模型)

class PrototypeManager:
    def __init__(self, num_classes, feature_dim, ema_alpha=0.9, device='cuda'):
        """
        Args:
            ema_alpha: 历史原型的保留比例 (0.9 表示新中心只占 0.1 权重)
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.ema_alpha = ema_alpha
        self.device = device
        
        # 初始化原型 (N_class, Dim)
        self.prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.is_initialized = False

    def update(self, features, reliable_mask, reliable_labels):
        """
        利用当前的可靠样本更新原型
        features: [N, Dim]
        reliable_mask: [N] bool
        reliable_labels: [N] (可能是伪标签或干净标签，取决于传入什么)
        """
        features = features.detach()
        # 确保特征归一化 (配合余弦相似度)
        features = F.normalize(features, dim=1)
        
        # 筛选可靠样本
        rel_feats = features[reliable_mask]
        rel_targets = reliable_labels[reliable_mask]
        
        if len(rel_feats) == 0:
            return

        # 计算当前 Batch/Epoch 的新类中心
        new_protos = torch.zeros_like(self.prototypes)
        
        # 这种写法比循环快
        # Numerator: Sum features per class
        # Denominator: Count per class
        one_hot = F.one_hot(rel_targets.long(), self.num_classes).float() # [N_rel, C]
        
        # [C, N_rel] @ [N_rel, Dim] -> [C, Dim]
        sum_features = torch.mm(one_hot.T, rel_feats) 
        counts = one_hot.sum(dim=0).unsqueeze(1) + 1e-8
        
        current_means = sum_features / counts
        current_means = F.normalize(current_means, dim=1) # 再次归一化

        # EMA 更新
        if not self.is_initialized:
            self.prototypes = current_means
            self.is_initialized = True
        else:
            # 只有当前 batch 出现过的类别才更新，没出现的保持原样
            # mask: [C, 1]
            active_classes = (one_hot.sum(dim=0).unsqueeze(1) > 0).float()
            
            updated_protos = self.ema_alpha * self.prototypes + (1 - self.ema_alpha) * current_means
            
            # 组合：活跃类用更新的，不活跃类用旧的
            self.prototypes = active_classes * updated_protos + (1 - active_classes) * self.prototypes
            
        # 保持原型在单位球面上
        self.prototypes = F.normalize(self.prototypes, dim=1)

    def predict(self, query_features):
        """
        基于余弦相似度进行预测
        return: 
            sims: [N, C] 相似度分数
            preds: [N] 预测类别
        """
        query_features = F.normalize(query_features, dim=1)
        # [N, Dim] @ [Dim, C] -> [N, C]
        sims = torch.mm(query_features, self.prototypes.T)
        preds = sims.argmax(dim=1)
        return sims, preds


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
    else: # 默认为 CIFAR
        mean, std = ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]) if '100' in dataset_name else ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    # --- (修改) 扩展 Transform 逻辑 ---
    
    if dataset_name == 'Treeversity':
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
def get_base_encoder(network_name, dataset_name):
    
    
    # --- (修改) 扩展使用预训练权重的条件 ---
    # use_pretrained = dataset_name in ['Treeversity', 'Benthic', 'Plankton','Synthetic',]
    use_pretrained = dataset_name in ['Treeversity', 'Benthic', 'Plankton']
    
    if network_name == 'R50':
        base_model = resnet50(weights='IMAGENET1K_V1' if use_pretrained else None)
    else: # Default to R18
        base_model = resnet18(weights='IMAGENET1K_V1' if use_pretrained else None)

    feature_dim = base_model.fc.in_features
    
    # if 'CIFAR' in dataset_name:
    if 'CIFAR' in dataset_name or 'Synthetic' in dataset_name:
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        
    encoder = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten())
    return encoder, feature_dim


## --- 🚀 MODIFICATION START: Removed SimSiam class --- ##
# class SimSiam(nn.Module): ... (Class Removed)
# def simsiam_loss_fn(p, z): ... (Function Removed)
## --- MODIFICATION END --- ##


class FeatureExtractionDataset(Dataset):
    def __init__(self, base_dataset, weak_t, strong_t): 
        self.base_dataset = base_dataset
        self.weak_t = weak_t
        self.strong_t = strong_t
        
        # --- (修改) ---
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
        # --- (修改结束) ---

    def __len__(self): 
        return len(self.base_dataset)
        
    def __getitem__(self, index):
        # 1. 获取原始图像
        
        # --- (修改) ---
        if self.is_crowd:
            # Crowdsource: .data 是 'list' of paths, 可以直接用 [index]
            img_path = self.base_dataset.data[index]
            img = Image.open(img_path).convert('RGB')

        else:
            # CIFAR: .data 是 numpy 数组
            img = Image.fromarray(self.base_dataset.data[index])
        # --- (修改结束) ---

        weak_img = self.weak_t(img)
        if self.strong_t is None:
            return weak_img, index
        return (weak_img, self.strong_t(img)), index


# ==============================================================================
#      Section 2.5: 统一数据集 + 动态重要性采样器 (Unified Dataset + Importance Sampler)
# ==============================================================================

class ImageOnlyDataset(Dataset):
    def __init__(self, base_dataset, weak_t, strong_t, index_map=None):
        self.base_dataset = base_dataset
        self.weak_t = weak_t
        self.strong_t = strong_t
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
        self.index_map = list(range(len(base_dataset))) if index_map is None else list(index_map)
        
    def __len__(self):
        return len(self.index_map)

    def update_index_map(self, new_index_map):
        self.index_map = list(new_index_map)
        
    def __getitem__(self, idx):
        original_idx = self.index_map[idx]
        if self.is_crowd:
            img_path = self.base_dataset.data[original_idx]
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.fromarray(self.base_dataset.data[original_idx])
        return self.weak_t(img), self.strong_t(img), original_idx

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
            base_dataset: 原始数据集 (CIFAR/Crowdsource)
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
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        original_idx, label, is_reliable = self.data_list[idx]
        
        # 获取原始图像
        if self.is_crowd:
            img_path = self.base_dataset.data[original_idx]
            img = Image.open(img_path).convert('RGB')
        else:  # CIFAR
            img = Image.fromarray(self.base_dataset.data[original_idx])
        
        return (self.weak_t(img), self.strong_t(img),
                label, is_reliable, original_idx)


class TemporalStateManager:
    def __init__(self, num_samples, num_classes, max_epochs, history_len=5, use_disambiguation=True):
        self.N, self.C = num_samples, num_classes
        self.max_epochs = max_epochs
        self.history_len = history_len
        self.use_disambiguation = use_disambiguation 
        # 🚀 三方队列共识硬标签快照队列
        # 存储格式:达成三方共识存入 Label(0-99),未达成存入 -1
        self.tri_consensus_history = deque(maxlen=history_len)
        
        # 为了判断 "始终不可靠",保留可靠性历史
        self.is_reliable_history = deque(maxlen=history_len)
        # 队列维护:记录每个样本的可靠性状态
        # self.is_reliable_history = deque(maxlen=history_len)
        
        # 记录 1: 剪枝后的 KNN 标签 (Topology-KNN)
        self.pruned_pl_history = deque(maxlen=history_len)
        
        # 记录 2: 基于模型预测的几何标签 (Model-KNN)
        self.geo_pl_history = deque(maxlen=history_len)
        
        # 🚀 记录 3: [新增] 基于类原型的预测标签 (Proto-PL)
        self.proto_pl_history = deque(maxlen=history_len)
        
        # 消歧参考:模型历史预测分布的移动平均 (EMA)
        self.prob_ema = torch.ones(num_samples, num_classes) / num_classes
        self.ema_m = 0.995 

    def update_ema(self, current_model_probs):
        """使用模型预测更新 EMA"""
        self.prob_ema = self.ema_m * self.prob_ema + (1 - self.ema_m) * current_model_probs.cpu()

    def update_history(self, is_reliable_mask, pruned_pl, geo_pl=None, proto_pl=None):
        """
        存入历史轨迹 
        Args:
            is_reliable_mask: 当前 epoch 是否被选为可靠
            pruned_pl: Phase 2 产生的 KNN 伪标签
            geo_pl: 基于去自身化邻居的模型几何预测标签
            proto_pl: [新增] 基于类原型的预测标签
        """
        self.is_reliable_history.append(is_reliable_mask.cpu().bool())
        self.pruned_pl_history.append(pruned_pl.cpu().long())
        
        if geo_pl is not None:
            self.geo_pl_history.append(geo_pl.cpu().long())
        else:
            self.geo_pl_history.append(torch.full_like(pruned_pl, -1).cpu().long())

        # 🚀 [新增] 记录 Proto 历史
        if proto_pl is not None:
            self.proto_pl_history.append(proto_pl.cpu().long())
        else:
            self.proto_pl_history.append(torch.full_like(pruned_pl, -1).cpu().long())

    def get_dynamic_disambiguation(self, epoch, device):
        if not self.use_disambiguation or epoch == 0:
            return torch.ones(self.N, self.C).to(device)
            
        alpha = (epoch / self.max_epochs) ** 2
        D = torch.pow(self.prob_ema + 1e-12, alpha)
        return D.to(device)

    def get_salvage_mask(self):
        """
        Tri-Consensus Salvage (三方共识打捞)
        要求: KNN (Topology), Model-Geo, 和 Prototype 在历史上都稳定且达成一致。
        """
        # 1. Base safety check
        curr_len = len(self.is_reliable_history)
        if curr_len == 0:
            return None, None, None, None

        # 2. Stack History
        rel_stack = torch.stack(list(self.is_reliable_history)) 
        pl_stack = torch.stack(list(self.pruned_pl_history))

        # 3. Calculate Metrics
        # A. Always Unreliable (始终是弃儿)
        always_unreliable = (rel_stack.sum(dim=0) == 0)

        # B. KNN Consistency (KNN 自身历史稳定)
        knn_consistency = (pl_stack == pl_stack[-1:]).all(dim=0)

        # C. Geo Consistency (Model-KNN 自身历史稳定)
        if len(self.geo_pl_history) > 0:
            geo_stack = torch.stack(list(self.geo_pl_history))
            valid_geo = (geo_stack != -1).all(dim=0)
            geo_consistency = (geo_stack == geo_stack[-1:]).all(dim=0) & valid_geo
        else:
            geo_consistency = torch.zeros_like(knn_consistency, dtype=torch.bool)
            geo_stack = pl_stack # Fallback

        # 🚀 D. [新增] Proto Consistency (Proto 自身历史稳定)
        if len(self.proto_pl_history) > 0:
            proto_stack = torch.stack(list(self.proto_pl_history))
            valid_proto = (proto_stack != -1).all(dim=0)
            proto_consistency = (proto_stack == proto_stack[-1:]).all(dim=0) & valid_proto
        else:
            # 如果没有 proto 历史,暂时放宽此条件或设为 False (视严格程度而定)
            # 建议: 如果启用了 Proto,这里应该是 False;为了兼容性先设 False
            proto_consistency = torch.zeros_like(knn_consistency, dtype=torch.bool)
            proto_stack = pl_stack # Fallback

        # F. Cross-Track Agreement(三方:KNN / Geo / Proto)
        cross_track_agreement = (pl_stack[-1] == geo_stack[-1]) & (geo_stack[-1] == proto_stack[-1])

        # 4. Source Attribution (归因分析,用于日志)
        knn_src_mask = always_unreliable & knn_consistency
        geo_src_mask = always_unreliable & geo_consistency

        # 5. Combine Masks (Strict Intersection)
        # 必须: 始终不可靠 AND 历史稳定 AND 三方意见一致
        salvage_mask = knn_src_mask & geo_src_mask & proto_consistency & cross_track_agreement

        # 6. Determine Labels
        # 既然三方一致,直接取任意一个(这里取 KNN)的标签即可
        final_salvaged_labels = self.pruned_pl_history[-1].clone()

        return salvage_mask, final_salvaged_labels, knn_src_mask, geo_src_mask


    def update_tri_consensus(self, is_reliable_mask, p_model, p_knn, p_proto):
        """计算并存入瞬时三方共识快照（Model / KNN / Proto）。

        注意：`is_reliable_history` 的维护在 `update_history(...)` 中完成，
        这里不再重复 append，避免出现"一个 epoch 写两次历史"的现象。
        """
        # 获取各方硬预测
        pl_m = p_model.argmax(dim=1).cpu()
        pl_k = p_knn.argmax(dim=1).cpu()
        pl_p = p_proto.argmax(dim=1).cpu()

        tri_mask = (pl_m == pl_k) & (pl_k == pl_p)

        # 构造快照:共识则留标签,否则 -1
        snapshot = torch.where(tri_mask, pl_m, torch.tensor(-1))
        self.tri_consensus_history.append(snapshot)


    def get_stable_tri_mask(self):
        """筛选出：

        1. 历史区间内从未进入可靠集（针对打捞场景）
        2. 在 history_len 内每一帧都达成三方共识（无 -1）
        3. 标签在 history_len 内完全锁定
        """
        if len(self.tri_consensus_history) < self.history_len or len(self.is_reliable_history) < self.history_len:
            return None, None

        rel_stack = torch.stack(list(self.is_reliable_history))
        tri_stack = torch.stack(list(self.tri_consensus_history))

        always_unreliable = (rel_stack.sum(dim=0) == 0)

        ever_disagree = (tri_stack == -1).any(dim=0)
        always_tri_agree = ~ever_disagree

        latest_label = tri_stack[-1]
        label_is_locked = (tri_stack == latest_label).all(dim=0)

        stable_mask = always_unreliable & always_tri_agree & label_is_locked
        return stable_mask, latest_label

def get_topology_guided_affinity(raw_D, neighbors_indices, current_soft_labels, num_classes,
                                  rel_mode='masked_entropy', gamma=2.0, eps=1e-12,
                                  kl_self_mode='with_self'):
    """
    修改来源: 结合投影_众包_自节点_模型预测融合.py -> 结合投影_众包_自节点_模型预测融合_kl消融.py
    修改位置: get_topology_guided_affinity() 函数签名及 kl 分支
    修改内容:
      - 新增参数 kl_self_mode: 控制 kl 模式下 p_knn 是否包含自身节点 (第0列)
        * with_self (default): 原始行为，p_knn 使用完整 [N, K+1] 矩阵
        * no_self: 仅在 kl 分支中，用 [:, 1:] 切片排除自身，p_knn 基于纯邻居
      - masked_entropy / agree 分支不受影响
    Modified: Added kl_self_mode to control whether self-node is included in kl branch p_knn.
    """
    N, K_plus_1 = neighbors_indices.shape

    # --- Step 1: 纯线性平滑 KNN 估计 (包含自身) ---
    linear_weights = raw_D # [修改点] 移除 [:, 1:]
    linear_weights = linear_weights / (linear_weights.sum(dim=1, keepdim=True) + eps)

    neighbor_labels = F.embedding(neighbors_indices, current_soft_labels)  # [修改点] 移除 [:, 1:]
    knn_scores_smooth = (linear_weights.unsqueeze(-1) * neighbor_labels).sum(dim=1)
    p_knn = knn_scores_smooth / (knn_scores_smooth.sum(dim=1, keepdim=True) + eps)

    # --- Step 2: 计算节点信誉度分数 ---
    p_self = current_soft_labels / (current_soft_labels.sum(dim=1, keepdim=True) + eps)
    if rel_mode == 'masked_entropy':
        masked_scores = p_knn * p_self
        masked_prob = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + eps)
        norm_score = -torch.sum(masked_prob * torch.log(masked_prob + eps), dim=1) / (np.log(num_classes) + eps)
    elif rel_mode == 'kl':
        # ===========================================================================
        # [消融修改 / Ablation Modified] kl_self_mode 控制 p_knn 是否排除自身节点
        # 修改来源: 结合投影_众包_自节点_模型预测融合.py get_topology_guided_affinity() kl 分支
        # 修改内容: 新增 no_self 分支，对 neighbors_indices[:, 1:] 和 raw_D[:, 1:] 单独
        #           计算纯邻居 p_knn_for_kl，仅影响 KL 可靠性分数，不影响最终亲和矩阵
        # Modified: Added no_self branch to compute p_knn from pure neighbors (excluding self)
        #           for KL reliability scoring only; final affinity matrix is unaffected.
        # ===========================================================================
        if kl_self_mode == 'no_self' and K_plus_1 > 1:
            # 排除自身节点（索引0），仅使用纯邻居 [:, 1:]
            # Exclude self-node (index 0), use pure neighbors [:, 1:] only
            nei_D = raw_D[:, 1:]        # [N, K]
            nei_idx = neighbors_indices[:, 1:]  # [N, K]
            nei_w = nei_D / (nei_D.sum(dim=1, keepdim=True) + eps)
            nei_labels = F.embedding(nei_idx, current_soft_labels)
            nei_agg = (nei_w.unsqueeze(-1) * nei_labels).sum(dim=1)
            p_knn_for_kl = nei_agg / (nei_agg.sum(dim=1, keepdim=True) + eps)
        else:
            # with_self: 原始行为，p_knn 包含自身节点
            # with_self: original behavior, p_knn includes self-node
            p_knn_for_kl = p_knn
        cross_entropy = -torch.sum(p_self * torch.log(p_knn_for_kl + eps), dim=1)
        norm_score = cross_entropy / (np.log(num_classes) + eps)
    elif rel_mode == 'agree':
        agree_mass = torch.sum(p_knn * p_self, dim=1).clamp(min=eps, max=1.0)
        norm_score = (-torch.log(agree_mass)) / (np.log(num_classes) + eps)
    else:
        raise ValueError(f"Unknown rel_mode: {rel_mode}")

    gamma = float(gamma)
    reliability_scores = torch.exp(-gamma * (norm_score ** 2)) 

    # --- Step 3: 生成最终亲和矩阵 (始终使用完整 [N, K+1] 矩阵) ---
    # Final affinity matrix always uses the full [N, K+1] matrix.
    reliability_scores_expanded = reliability_scores.unsqueeze(1) 
    all_reliabilities = F.embedding(neighbors_indices, reliability_scores_expanded).squeeze(-1)
    refined_sim = raw_D * all_reliabilities

    return refined_sim, reliability_scores

def get_topology_daes_affinity(raw_D, neighbors_indices, current_soft_labels, args):
    """
    修改来源: 结合投影_众包_自节点_模型预测融合.py -> 结合投影_众包_自节点_模型预测融合_kl消融.py
    修改位置: get_topology_daes_affinity() 函数 kl 分支
    修改内容: 读取 args.kl_self_mode，在 kl 模式下可选是否排除自身节点进行 KL 可靠性计算
              DAES 第二阶段（邻域熵计算）始终使用完整矩阵，不受 kl_self_mode 影响。
    Modified: Reads args.kl_self_mode; in kl mode, optionally excludes self-node for KL
              reliability. DAES stage 2 (neighborhood entropy) always uses full matrix.
    """
    eps = getattr(args, 'topology_rel_eps', 1e-12)
    N, K_plus_1 = neighbors_indices.shape
    num_classes = args.num_classes

    # 第一阶段：Topology（始终使用完整矩阵计算 p_knn）
    # Stage 1: Topology (always compute p_knn from full matrix)
    linear_weights = raw_D / (raw_D.sum(dim=1, keepdim=True) + eps) # [修改点]
    neighbor_labels_for_top = F.embedding(neighbors_indices, current_soft_labels) # [修改点]
    knn_scores_smooth = (linear_weights.unsqueeze(-1) * neighbor_labels_for_top).sum(dim=1)
    p_knn = knn_scores_smooth / (knn_scores_smooth.sum(dim=1, keepdim=True) + eps)

    p_self = current_soft_labels / (current_soft_labels.sum(dim=1, keepdim=True) + eps)
    rel_mode = getattr(args, 'topology_rel_mode', 'masked_entropy')
    
    if rel_mode == 'masked_entropy':
        masked_scores = p_knn * p_self
        masked_prob = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + eps)
        norm_score = -torch.sum(masked_prob * torch.log(masked_prob + eps), dim=1) / (np.log(num_classes) + eps)
    elif rel_mode == 'kl':
        # ===========================================================================
        # [消融修改 / Ablation Modified] kl_self_mode 控制 kl 矩阵自身节点
        # 修改来源: 结合投影_众包_自节点_模型预测融合.py get_topology_daes_affinity() kl 分支
        # 修改内容: 新增 no_self 模式，对纯邻居 [:, 1:] 单独聚合得到 p_knn_for_kl
        #           仅影响 KL 可靠性得分，不影响 DAES 阶段的完整矩阵计算
        # Modified: no_self mode aggregates only pure neighbors for KL reliability scoring.
        #           DAES neighborhood entropy computation is unaffected.
        # ===========================================================================
        kl_self_mode = getattr(args, 'kl_self_mode', 'with_self')
        if kl_self_mode == 'no_self' and K_plus_1 > 1:
            # 排除自身节点，仅用纯邻居计算 KL 可靠性
            # Exclude self-node; compute KL reliability from pure neighbors only
            nei_D = raw_D[:, 1:]        # [N, K]
            nei_idx = neighbors_indices[:, 1:]  # [N, K]
            nei_w = nei_D / (nei_D.sum(dim=1, keepdim=True) + eps)
            nei_labels = F.embedding(nei_idx, current_soft_labels)
            nei_agg = (nei_w.unsqueeze(-1) * nei_labels).sum(dim=1)
            p_knn_for_kl = nei_agg / (nei_agg.sum(dim=1, keepdim=True) + eps)
        else:
            # with_self: 原始行为，p_knn 包含自身节点
            # with_self: original behavior, p_knn includes self-node
            p_knn_for_kl = p_knn
        norm_score = -torch.sum(p_self * torch.log(p_knn_for_kl + eps), dim=1) / (np.log(num_classes) + eps)
    elif rel_mode == 'agree':
        agree_mass = torch.sum(p_knn * p_self, dim=1).clamp(min=eps, max=1.0)
        norm_score = (-torch.log(agree_mass)) / (np.log(num_classes) + eps)

    gamma = float(getattr(args, 'topology_rel_gamma', 2.0))
    reliability_scores = torch.exp(-gamma * (norm_score ** 2))

    # 第二阶段：DAES（始终使用完整矩阵，不受 kl_self_mode 影响）
    # Stage 2: DAES (always uses full matrix, unaffected by kl_self_mode)
    att_temp = getattr(args, 'daes_spatial_temp', 0.5)
    base_tau = getattr(args, 'daes_base_tau', 0.1)
    entropy_coeff = getattr(args, 'daes_entropy_coeff', 0.5)
    sim_power = getattr(args, 'daes_sim_power', 2.0)
    
    # [修改点] 移除切片，直接使用完整矩阵评估邻域熵
    spatial_weights = F.softmax(raw_D / att_temp, dim=1).unsqueeze(-1)
    neighbor_labels = F.embedding(neighbors_indices, current_soft_labels)
    
    local_mean = (neighbor_labels * spatial_weights).sum(dim=1)
    local_entropy = -torch.sum(local_mean * torch.log(local_mean + eps), dim=1)
    norm_entropy = local_entropy / np.log(num_classes)
    
    tau_dynamic = (base_tau + (torch.pow(norm_entropy, 2) * entropy_coeff)).unsqueeze(1)

    # 第三阶段：融合（始终使用完整矩阵）
    # Stage 3: Fusion (always uses full matrix)
    scaled_sim = torch.pow(raw_D, sim_power) / tau_dynamic
    max_val, _ = scaled_sim.max(dim=1, keepdim=True)
    daes_weights = torch.exp(scaled_sim - max_val.detach())
    
    neighbor_reliabilities = F.embedding(neighbors_indices, reliability_scores.unsqueeze(1)).squeeze(-1)
    final_neighbor_weights = daes_weights * neighbor_reliabilities
    
    return final_neighbor_weights



import torch
import torch.nn.functional as F
import numpy as np

def reliable_pseudolabel_selection_advanced(logger, args, device, trainloader, features, epoch,
                                            state_manager, model_preds=None, proto_manager=None):
    """
    博士级增强版：双源感知可靠集筛选 (支持众包先验软投影与 PLL 硬掩码分离)
    仅执行两次迭代 (Iteration 0 和 Iteration 1)
    """
    N = features.shape[0]
    dataset = trainloader.dataset
    eps_stable = 1e-8

    # 获取原始静态约束 (PLL 硬掩码)
    if hasattr(dataset, 'original_soft_labels'):
        static_cand_mask = torch.tensor(dataset.original_soft_labels, device=device, dtype=torch.float64)
    else:
        static_cand_mask = torch.tensor(dataset.soft_labels, device=device, dtype=torch.float64)

    # 🌟 获取众包真实权重先验 (Crowdsource Soft Prior)
    is_crowd = getattr(args, 'dataset', '') in ['Treeversity', 'Benthic', 'Plankton']
    crowd_prior = None
    if is_crowd and hasattr(dataset, 'weights'):
        crowd_prior = torch.tensor(dataset.weights, device=device, dtype=torch.float32) + eps_stable

    # 获取当前动态起点与干净标签
    current_fixed_labels = torch.tensor(dataset.soft_labels, device=device).float()
    clean_labels = torch.tensor(dataset.clean_labels, device=device, dtype=torch.long)

    # ==============================================================================
    # 核心辅助逻辑:双源统一筛选闸门
    # ==============================================================================
    def _filter_logic(soft_probs):
        prob_temp = torch.clamp(soft_probs, min=1e-6, max=1-1e-6)
        discrepancy = -torch.log(prob_temp)
        max_p, max_idx = soft_probs.max(dim=1)

        # 准入判定：无论是硬投影还是软投影，最终预测的最大值必须在原分布的非零支撑集内
        in_static_cand = (static_cand_mask.gather(1, max_idx.unsqueeze(1)).squeeze(1) > 0)

        total_cand_mask = in_static_cand
        rel_mask = torch.zeros(N, device=device)
        counts = torch.bincount(max_idx[total_cand_mask], minlength=args.num_classes).double()
        limit = torch.quantile(counts, args.delta) if counts.numel() > 0 else 0

        # SRSE reliable-set rule: class-balanced quota over source-supported
        # predictions, ranked by classwise discrepancy within each predicted class.
        for i in range(args.num_classes):
            idx_c_mask = total_cand_mask & (max_idx == i)
            if idx_c_mask.sum() == 0: continue

            k_val = min(limit.item(), idx_c_mask.sum().float().item())
            if k_val < 1: continue

            _, top_idx = torch.topk(discrepancy[idx_c_mask, i], k=int(k_val), largest=False)
            rel_mask[idx_c_mask.nonzero().squeeze(1)[top_idx]] = 1.0

        n_selected = rel_mask.sum().item()
        acc = (max_idx[rel_mask.bool()] == clean_labels[rel_mask.bool()]).float().mean().item() if n_selected > 0 else 0.0
        return rel_mask, max_idx, acc, n_selected

    # 1. 拓扑构建
    D_mh, neighbors_mh = knn_search_pytorch_chunked(features, args.k_val, num_heads=args.knn_heads)
    raw_sim = F.relu(D_mh).float()

    # [MODIFIED - v2 gitdiffer] 众包数据集使用 weights 作为 KNN 起点
    if crowd_prior is not None:
        curr_soft_out = crowd_prior.clone()
        logger.info(f"✨ [KNN Input] Using crowd_prior (weights) as 1st KNN input for crowdsource dataset")
    else:
        curr_soft_out = current_fixed_labels.clone()

    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 [Epoch {epoch+1}] Reliable Selection - Stage Breakdown")
    logger.info(f"{'='*80}")

    with torch.no_grad():
        mask_init, pred_init, acc_init, size_init = _filter_logic(curr_soft_out)
        overall_pred_init = curr_soft_out.argmax(dim=1)
        overall_acc_init = (overall_pred_init == clean_labels).float().mean().item()
        logger.info(f"📊 [Stage 0] Initial (Before Propagation)")
        logger.info(f"   ├─ Selected: {int(size_init)} samples | Acc: {acc_init*100:.2f}% | Overall Acc: {overall_acc_init*100:.2f}%")

    # ==============================================================================
    # 第一次迭代 (i = 0): 基于初始输入的 KNN 聚合
    # ==============================================================================
    propagation_input_1 = curr_soft_out

    refined_sim_1 = get_weight_matrix(args.sim_mode_1, raw_sim, neighbors_mh, propagation_input_1, args)

    voting_weights_1 = refined_sim_1
    neighbor_vals_1 = propagation_input_1[neighbors_mh]
    weighted_votes_1 = torch.einsum('nk,nkc->nc', voting_weights_1, neighbor_vals_1)
    curr_soft_out = F.softmax(weighted_votes_1, dim=1)

    with torch.no_grad():
        mask_st1, pred_st1, acc_st1, size_st1 = _filter_logic(curr_soft_out)
        overall_pred_st1 = curr_soft_out.argmax(dim=1)
        overall_acc_st1 = (overall_pred_st1 == clean_labels).float().mean().item()
        logger.info(f"📊 [Stage 1] After 1st Propagation")
        logger.info(f"   ├─ Selected: {int(size_st1)} samples | Acc: {acc_st1*100:.2f}% | Overall Acc: {overall_acc_st1*100:.2f}%")

    # ==============================================================================
    # [MODIFIED - bayes unified variant] 统一贝叶斯证据融合 / Unified Bayesian Evidential Fusion
    # 所有数据集共用同一条 refine 路径：
    #   prior_effective = normalize(omega ⊙ p_model_effective)
    #   propagation_input_2 = normalize(r_i * p_knn1 + (1-r_i) * prior_effective)
    # omega 只约束模型分支，图传播分支仍可通过邻域证据恢复源先验缺失类别。
    # ==============================================================================
    p_knn1 = curr_soft_out  # Stage 1 结果 [N, C]
    _dataset_name = getattr(args, 'dataset', '')
    _is_crowd = _dataset_name in ['Treeversity', 'Benthic', 'Plankton']

    if _is_crowd:
        logger.info("✨ [Refine Branch] Crowdsource dataset enters unified Bayesian evidential fusion path")
    else:
        logger.info("✨ [Refine Branch] CIFAR dataset enters unified Bayesian evidential fusion path")

    # [直接计算 r_i, 基于 KNN 和模型置信度] — 遵循用户公式
    adap_eps  = float(getattr(args, 'adap_rel_eps', 1e-12))
    # adap_gamma = float(getattr(args, 'adap_rel_gamma', 2.0))
    # ce_direction = getattr(args, 'adap_ce_direction', 'knn_over_model')
    # num_classes = args.num_classes

    model_warmup_epochs = float(getattr(args, 'model_warmup_epochs', 10))
    max_w_model_val = float(getattr(args, 'max_w_model', 0.5))
    w_model = max_w_model_val * min(1.0, epoch / max(model_warmup_epochs, 1.0))

    if model_preds is not None:
        p_model_raw = model_preds.float()
        # Progressive Fusion 保持模型权重的平滑演进
        p_model_effective = w_model * p_model_raw + (1.0 - w_model) * p_knn1.detach()

        logger.info(f"✨ [Stage 2.5-pre] Progressive Fusion: w_model={w_model:.4f} "
                    f"(epoch={epoch}, warmup={int(model_warmup_epochs)})")

        # --- 🚀 [NEW r_i formula] ---
        # conf_knn = max(P_knn), conf_model = max(P_model_fusion)
        conf_knn = p_knn1.max(dim=1)[0]
        conf_model = p_model_raw.max(dim=1)[0]

        # r_i = conf_knn / (conf_knn + conf_model + eps)
        r_i = (conf_knn / (conf_knn + conf_model + adap_eps)).unsqueeze(1)
        # -----------------------------
    else:
        p_model_effective = p_knn1.clone()
        p_model_raw = p_knn1.clone()
        w_model = 0.0
        r_i = torch.ones(N, 1, device=p_knn1.device)

    with torch.no_grad():
        logger.info(f"📊 [Stage 2.5] Reliability Scores (Confidence-based Ratio)")
        logger.info(f"   ├─ r_i: mean={r_i.mean().item():.4f}  std={r_i.std().item():.4f}  "
                    f"min={r_i.min().item():.4f}  max={r_i.max().item():.4f}")

    omega = crowd_prior if crowd_prior is not None else static_cand_mask.float()
    force_topology_only = bool(getattr(args, 'force_old_branch', False))
    if force_topology_only and _dataset_name in ['Treeversity', 'Benthic', 'Plankton']:
        r_i = torch.ones_like(r_i)
        logger.info(f"✨ [Refine Branch] force_old_branch=True -> r_i forced to 1.0 for {_dataset_name}")

    # 保留数据先验 (omega) 对模型预测的约束
    prior_effective  =  (p_model_effective * omega) / ((p_model_effective * omega).sum(dim=1, keepdim=True) + adap_eps)

    # fused_blend = r_i * P_knn + (1 - r_i) * prior_effective
    fused_blend = r_i * p_knn1 + (1.0 - r_i) * prior_effective
    propagation_input_2 = fused_blend / (fused_blend.sum(dim=1, keepdim=True) + adap_eps)

    with torch.no_grad():
        mask_s28, pred_s28, acc_s28, size_s28 = _filter_logic(propagation_input_2)
        overall_acc_s28 = (propagation_input_2.argmax(dim=1) == clean_labels).float().mean().item()
        logger.info(f"📊 [Stage 2.8] Unified Bayesian Fusion (Confidence-based r_i)")
        logger.info(f"   ├─ omega support: mean={(omega > 0).float().sum(dim=1).float().mean().item():.2f} active classes/sample")
        logger.info(f"   ├─ Selected: {int(size_s28)} samples | Acc: {acc_s28*100:.2f}% | Overall Acc: {overall_acc_s28*100:.2f}%")

    # Stage 2: 第二次 KNN 传播 (最终) — 与旧版一致，总计 2 次
    refined_sim_2 = get_weight_matrix(args.sim_mode_2, raw_sim, neighbors_mh, propagation_input_2, args)
    neighbor_vals_2 = propagation_input_2[neighbors_mh]
    weighted_votes_2 = torch.einsum('nk,nkc->nc', refined_sim_2, neighbor_vals_2)
    curr_soft_out = F.softmax(weighted_votes_2, dim=1)

    mask_final, pred_final, acc_rel_final, size_rel_final = _filter_logic(curr_soft_out)
    mask_final = mask_final.float()

    overall_pred_final = curr_soft_out.argmax(dim=1)
    overall_acc_final = (overall_pred_final == clean_labels).float().mean().item()

    logger.info(f"\U0001f4ca [Stage 2] Final Selection (After 2nd Propagation)")
    logger.info(f"   \u251c\u2500 Selected: {int(size_rel_final)} samples | Acc: {acc_rel_final*100:.2f}% | Overall Acc: {overall_acc_final*100:.2f}%")

    logger.info(f"{'='*80}\n")

    with torch.no_grad():
        # [注意] 这里的扩展维度应取决于你的 neighbors_mh 的第二维度的实际大小 (包含自身的话通常是 k+1)
        mask_expanded = mask_final.view(-1, 1).expand(-1, args.k_val + 1).float()
        w_pruned = raw_sim * torch.gather(mask_expanded, 0, neighbors_mh)

        w_p_sum = w_pruned.sum(dim=1)
        w_p_norm = w_pruned / (w_p_sum.unsqueeze(1) + 1e-12)
        soft_out_pruned = torch.sum(F.embedding(neighbors_mh, curr_soft_out) * w_p_norm.view(N, -1, 1), dim=1)
        pruned_pl = soft_out_pruned.argmax(dim=1)

    model_hard_labels = model_preds.argmax(dim=1) if model_preds is not None else p_model_effective.argmax(dim=1)
    neighbor_model_labels = model_hard_labels[neighbors_mh]
    neighbor_model_onehot = F.one_hot(neighbor_model_labels, num_classes=args.num_classes).float()

    weights = raw_sim.unsqueeze(-1)
    geo_soft_out = torch.sum(neighbor_model_onehot * weights, dim=1)
    model_geo_pl = geo_soft_out.argmax(dim=1)

    if proto_manager is not None:
        _, proto_pl_all = proto_manager.predict(features)
    else:
        proto_pl_all = None

    state_manager.update_history(mask_final, pruned_pl, model_geo_pl, proto_pl=proto_pl_all)

    return mask_final.float(), pred_final, pred_final, pred_final, curr_soft_out.float()


@torch.no_grad()
def get_features(encoder, classifier, loader, device, feature_extract_view='weak_only'):
    encoder.eval(); classifier.eval(); all_features, all_predictions, all_indices = [], [], []
    for images_dual, indices in loader:
        if isinstance(images_dual, (tuple, list)):
            # Legacy path: a tuple of (weak_imgs, strong_imgs)
            weak_imgs = images_dual[0]
            strong_imgs = images_dual[1] if len(images_dual) > 1 else None
        else:
            # Weak-only feature-screening path.
            weak_imgs = images_dual
            strong_imgs = None
        weak_imgs = weak_imgs.to(device, non_blocking=True)

        with autocast():
            # SRSE screening uses weak-view features for KNN topology. The default
            # path also uses weak-view predictions to avoid a second global
            # strong-view forward before active-set construction.
            feat_w = encoder(weak_imgs)
            pred_w = F.softmax(classifier(feat_w), dim=1)

            if feature_extract_view == 'weak_strong_fusion':
                if strong_imgs is None:
                    raise ValueError('feature_extract_view=weak_strong_fusion requires strong-view batches.')
                strong_imgs = strong_imgs.to(device, non_blocking=True)
                feat_s = encoder(strong_imgs)
                pred_s = F.softmax(classifier(feat_s), dim=1)
                p_model = (pred_w + pred_s) / 2
            else:
                p_model = pred_w

        # 仍然使用弱增强视图的特征用于 KNN 拓扑构建
        all_features.append(F.normalize(feat_w.float()))
        all_predictions.append(p_model.float())
        all_indices.append(indices.cpu())

    all_features = torch.cat(all_features)
    all_predictions = torch.cat(all_predictions)
    all_indices = torch.cat(all_indices)

    return all_features[torch.argsort(all_indices)], all_predictions[torch.argsort(all_indices)]

class SoftMatchWeightManager:
    def __init__(self, num_samples, num_classes, n_sigma=2.0, momentum=0.99, device='cuda'): self.n_sigma, self.momentum, self.device = n_sigma, momentum, device; self.prob_model = torch.ones(num_samples, num_classes, device=device) / num_classes
    def __call__(self, preds, index, return_stats=False):
        self.prob_model[index] = self.momentum * self.prob_model[index] + (1 - self.momentum) * preds.detach(); max_probs_model = self.prob_model[index].max(dim=1)[0]; mu = max_probs_model.mean(); std = max_probs_model.std() if max_probs_model.size(0) > 1 else torch.tensor(1e-8, device=self.device); weights = torch.exp(-torch.pow(F.relu(mu - preds.max(dim=1)[0]), 2) / (2 * self.n_sigma * std**2 + 1e-8))
        return (weights.detach(), mu.item(), std.item()) if return_stats else weights.detach()











@torch.no_grad()
def evaluate(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval(); correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(): outputs = classifier(encoder(images))
        _, predicted = torch.max(outputs, 1); total += labels.size(0); correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0.0

# ==============================================================================
# 🛠️ Helper 1: 显存安全的高精度 KNN
# ==============================================================================
def knn_search_pytorch_chunked(feats, k, num_heads=1, chunk_size=4096):
    original_matmul_precision = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        N, D_dim = feats.shape
        if not feats.is_contiguous(): feats = feats.contiguous()

        if num_heads > 1:
            head_dim = D_dim // num_heads
            feats_ready = F.normalize(feats.view(N, num_heads, head_dim), p=2, dim=2).reshape(N, D_dim)
        else:
            feats_ready = F.normalize(feats, p=2, dim=1)

        final_sims, final_indices = [], []
        with torch.no_grad():
            database_t = feats_ready.t()
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                sim_matrix = torch.mm(feats_ready[i:end_idx], database_t)
                batch_sims, batch_indices = torch.topk(sim_matrix, k=min(k+1, N), dim=1, largest=True, sorted=True)
                final_sims.append(batch_sims); final_indices.append(batch_indices)
        return torch.cat(final_sims, dim=0) / float(num_heads), torch.cat(final_indices, dim=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_precision

def get_adaptive_affinity_matrix(raw_D, neighbors_indices, current_soft_labels, args):
    att_temp = getattr(args, 'daes_spatial_temp', 0.5)
    spatial_weights = F.softmax(raw_D / att_temp, dim=1).unsqueeze(-1)
    neighbor_labels = F.embedding(neighbors_indices, current_soft_labels)

    local_mean_raw = (neighbor_labels * spatial_weights).sum(dim=1)

    base_tau = getattr(args, 'daes_base_tau', 0.1)
    entropy_coeff = getattr(args, 'daes_entropy_coeff', 0.5)

    local_entropy = -torch.sum(local_mean_raw * torch.log(local_mean_raw + 1e-8), dim=1)
    norm_entropy = local_entropy / np.log(args.num_classes)
    tau_dynamic = (base_tau + (torch.pow(norm_entropy, 2) * entropy_coeff)).unsqueeze(1)

    sim_power = getattr(args, 'daes_sim_power', 2.0)
    scaled_sim = torch.pow(raw_D, sim_power) / tau_dynamic

    max_val, _ = scaled_sim.max(dim=1, keepdim=True)
    weights = torch.exp(scaled_sim - max_val.detach())

    return weights

def get_weight_matrix(mode, raw_D, neighbors_indices, ref_soft_labels, args):
    """
    修改点：直接处理完整的 [N, K+1] 矩阵，不再切片剔除自身节点。
    """
    if mode == 'daes':
        refined_D = get_adaptive_affinity_matrix(raw_D, neighbors_indices, ref_soft_labels, args)
    elif mode == 'topology':
        # 同步传入 kl_self_mode / Pass kl_self_mode to topology affinity
        refined_D, _ = get_topology_guided_affinity(
            raw_D, neighbors_indices, ref_soft_labels, args.num_classes,
            rel_mode=getattr(args, 'topology_rel_mode', 'masked_entropy'),
            gamma=getattr(args, 'topology_rel_gamma', 2.0),
            eps=getattr(args, 'topology_rel_eps', 1e-12),
            kl_self_mode=getattr(args, 'kl_self_mode', 'with_self'),
            support_ref=getattr(args, 'topology_support_ref', 2.0),
        )
    elif mode == 'topology_daes':
        refined_D = get_topology_daes_affinity(raw_D, neighbors_indices, ref_soft_labels, args)
    elif mode == 'exp':
        refined_D = torch.exp(raw_D / 0.1)
    else: # 'linear' or 'none'
        refined_D = raw_D

    return refined_D


def train_unified_single_stream(args, encoder, classifier, device,
                                unified_loader, optimizer,
                                logger, num_classes, global_labels,
                                global_is_reliable, proto_manager=None):
    """Train strictly on the epoch-local active set.

    `global_is_reliable` marks the active set A_t = reliable samples plus
    promoted salvage samples. Non-active samples must not enter this training
    path: no pseudo-target, no MixUp, no consistency/self-training loss.
    """
    encoder.train()
    classifier.train()
    scaler = GradScaler()

    total_loss_s = 0.0
    num_sup = 0
    active_batch_count = 0
    reliable_mixup_enabled = not (
        getattr(args, 'ablate_no_reliable_mixup', False)
        or getattr(args, 'no_reliable_mixup', False)
    )
    no_cr = bool(getattr(args, 'ablate_no_cr', False))

    for batch_data in unified_loader:
        weak_imgs, strong_imgs, indices = batch_data
        weak_imgs = weak_imgs.to(device, non_blocking=True)
        strong_imgs = strong_imgs.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        labels = global_labels[indices]
        active_mask = global_is_reliable[indices].bool()
        if active_mask.sum() == 0:
            continue

        optimizer.zero_grad()
        active_batch_count += 1

        rel_weak = weak_imgs[active_mask]
        rel_strong = strong_imgs[active_mask]
        rel_labels = labels[active_mask]

        if proto_manager is not None:
            with torch.no_grad():
                rel_feats_curr = encoder(rel_weak).detach()
                batch_rel_mask = torch.ones(rel_feats_curr.shape[0], device=device, dtype=torch.bool)
                proto_manager.update(rel_feats_curr, batch_rel_mask, rel_labels)

        s_labels = F.one_hot(rel_labels.long(), num_classes).float()
        s_labels = s_labels * (1 - args.lsr) + args.lsr / num_classes

        B_s = rel_weak.size(0)
        if reliable_mixup_enabled:
            perm_w = torch.randperm(B_s, device=device)
            lam_w = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            mix_w = lam_w * rel_weak + (1 - lam_w) * rel_weak[perm_w]
            mix_l_w = lam_w * s_labels + (1 - lam_w) * s_labels[perm_w]

            perm_s = torch.randperm(B_s, device=device)
            lam_s = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            mix_s = lam_s * rel_strong + (1 - lam_s) * rel_strong[perm_s]
            mix_l_s = lam_s * s_labels + (1 - lam_s) * s_labels[perm_s]
        else:
            mix_w = rel_weak
            mix_s = rel_strong
            mix_l_w = s_labels
            mix_l_s = s_labels

        with autocast():
            if no_cr:
                logits_s = classifier(encoder(mix_s))
                loss_s = -torch.sum(F.log_softmax(logits_s, 1) * mix_l_s, 1).mean()
            else:
                all_inputs = torch.cat([mix_w, mix_s])
                all_logits = classifier(encoder(all_inputs))
                logits_w = all_logits[:B_s]
                logits_s = all_logits[B_s:B_s * 2]

                loss_w = -torch.sum(F.log_softmax(logits_w, 1) * mix_l_w, 1).mean()
                loss_s_strong = -torch.sum(F.log_softmax(logits_s, 1) * mix_l_s, 1).mean()
                loss_s = (loss_w + loss_s_strong) * 0.5

        if loss_s.item() > 0 and not torch.isnan(loss_s):
            scaler.scale(loss_s).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss_s += loss_s.item() * B_s
            num_sup += B_s

    avg_loss_s = total_loss_s / num_sup if num_sup > 0 else 0
    if no_cr:
        logger.info(
            f"  -> [AblationCheck] cr=False | active_views=strong_only | "
            f"active_mixup={reliable_mixup_enabled} | NonActiveTrain=0"
        )
    logger.info(
        f"  -> [Train Loss] ActiveSup={avg_loss_s:.4f} | "
        f"ActiveBatches={active_batch_count} | NonActiveTrain=0."
    )

    return 0.0

def log_tri_consensus_diagnostics(logger, epoch, true_labels,
                                  unreliable_indices,
                                  model_pl, knn_pl, proto_pl,
                                  device):
    """诊断：Model vs KNN vs Prototype 在不可靠集上的多方博弈（三方版）。

    说明：该脚本不再把 EMA 预测作为共识参与方，因此这里也不再统计 EMA。
    """
    if len(unreliable_indices) == 0:
        return

    logger.info(f"📐 [Triangle Diagnostics] Epoch {epoch+1} | Unreliable Set: {len(unreliable_indices)}")

    idx = torch.tensor(unreliable_indices, device=device).long()
    target = true_labels[idx]

    # 获取预测
    p_model = model_pl[idx]
    p_knn   = knn_pl[idx]
    p_proto = proto_pl[idx]

    # 1) 单体准确率对比
    acc_model = (p_model == target).float().mean().item() * 100
    acc_knn   = (p_knn == target).float().mean().item() * 100
    acc_proto = (p_proto == target).float().mean().item() * 100
    logger.info(f"  ├─ 🎯 Individual Acc: Model={acc_model:.2f}% | KNN={acc_knn:.2f}% | Proto={acc_proto:.2f}%")

    # 2) 两两共识分析
    def analyze_pair(pred1, pred2):
        agree_mask = (pred1 == pred2)
        num_agree = agree_mask.sum().item()
        rate = num_agree / len(pred1) * 100
        if num_agree > 0:
            acc_consensus = (pred1[agree_mask] == target[agree_mask]).float().mean().item() * 100
        else:
            acc_consensus = 0.0
        return rate, acc_consensus

    rate_mk, acc_mk = analyze_pair(p_model, p_knn)
    rate_mp, acc_mp = analyze_pair(p_model, p_proto)
    rate_kp, acc_kp = analyze_pair(p_knn, p_proto)

    logger.info(f"  ├─ 🤝 Consensus Analysis (Agreement Rate / Consensus Acc):")
    logger.info(f"  │   ├─ Model & KNN:   Rate={rate_mk:.1f}% | Acc={acc_mk:.2f}%")
    logger.info(f"  │   ├─ Model & Proto: Rate={rate_mp:.1f}% | Acc={acc_mp:.2f}%")
    logger.info(f"  │   └─ KNN & Proto:   Rate={rate_kp:.1f}% | Acc={acc_kp:.2f}%")

    # 3) 三方共识
    grand_mask = (p_model == p_knn) & (p_knn == p_proto)
    num_grand = grand_mask.sum().item()
    if num_grand > 0:
        acc_grand = (p_model[grand_mask] == target[grand_mask]).float().mean().item() * 100
    else:
        acc_grand = 0.0
    logger.info(f"  └─ 🌟 Grand Consensus (All 3 agree): {num_grand} samples | Acc={acc_grand:.2f}%")
def run_single_experiment(args):
    start_time = time.time()
    set_seed(args.seed)

    # [EXP6/EXP8] 为避免不同实验互相覆盖输出目录,自动为 exp_name 增加后缀
    _exp_suffix = None
    if getattr(args, 'enable_knn1_geo_fuse', False):
        _exp_suffix = "_exp6_knn1_geofuse"
    if isinstance(_exp_suffix, str) and isinstance(args.exp_name, str) and (not args.exp_name.endswith(_exp_suffix)):
        args.exp_name = args.exp_name + _exp_suffix

    log_dir = os.path.join(args.out, args.exp_name, f"seed_{args.seed}")
    resume_checkpoint_path = _resolve_resume_checkpoint(args, log_dir)
    logger = setup_logger(log_dir, to_console=True, append=bool(resume_checkpoint_path))

    # 0. 本脚本仅运行第一阶段(不做蒸馏更新/标签覆写/系统重置)
    schedule_epochs = args.epochs
    max_run_epochs = int(getattr(args, 'max_run_epochs', 0) or 0)
    total_epochs = min(schedule_epochs, max_run_epochs) if max_run_epochs > 0 else schedule_epochs

    logger.info(f"--- Starting Dynamic Strategy Run with Seed: {args.seed} ---")
    logger.info(f"Settings: {vars(args)}")
    logger.info(f"📅 Schedule: Stage-1 only | Configured {schedule_epochs} eps | Running {total_epochs} eps")
    if resume_checkpoint_path:
        logger.info(f"🔁 Resume requested from checkpoint: {resume_checkpoint_path}")

    # 1. 实验类型判断 (WandB Grouping)
    exp_type = "Baseline_Full"
    if args.ablate_no_cr and args.no_reliable_mixup:
        exp_type = "No_CR_No_Rel_MixUp"
    elif args.ablate_no_cr:
        exp_type = "No_CR_StrongOnly"
    elif args.no_reliable_mixup:
        exp_type = "No_Rel_Mixup"

    # 2. WandB 初始化
    short_id = datetime.datetime.now().strftime('%H%M%S')
    run_name = f"{exp_type}_S{args.seed}_{short_id}"
    wandb.init(project="CIFAR100_Ablation_Study", name=run_name, group=f"{args.dataset}_{exp_type}", config=args, mode="disabled")

    device = torch.device(f"cuda:{args.cuda_dev}" if torch.cuda.is_available() else "cpu")
    args.seed_dataset = args.seed

    # 3. 数据加载与预处理
    num_classes_map = {'CIFAR10': 10, 'CIFAR100': 100, 'CIFAR100H': 100, 'Treeversity': 6, 'Benthic': 8, 'Plankton': 10, 'Synthetic': 6}
    num_classes = num_classes_map[args.dataset]
    args.num_classes = num_classes

    weak_t, strong_t, test_t = get_pals_transforms(args.dataset)

    # --- Dataset Loading Logic ---
    if args.dataset in ['CIFAR10', 'CIFAR100', 'CIFAR100H']:
        is_h = 'H' in args.dataset
        BaseClass = CIFAR100Partial if '100' in args.dataset else CIFAR10Partial
        base_train_ds = BaseClass(args, train=True, download=True, transform=None)

        # 初始化修改掩码
        if not hasattr(base_train_ds, 'modified_mask'):
            base_train_ds.modified_mask = np.zeros(len(base_train_ds), dtype=bool)

        if hasattr(base_train_ds, 'partial_noise'):
            logger.info(f"Generating simulated NPLL noise for {args.dataset} (pr={args.pr}, nr={args.nr})")
            if '100' in args.dataset:
                base_train_ds.partial_noise(args.pr, args.nr, heirarchical=is_h)
            else:
                base_train_ds.partial_noise(args.pr, args.nr)

        TestClass = datasets.CIFAR100 if '100' in args.dataset else datasets.CIFAR10
        test_ds = TestClass(root=args.train_root, train=False, download=True, transform=test_t)

    elif args.dataset in ['Treeversity', 'Benthic', 'Plankton']:
        lpi_args = argparse.Namespace(
            train_root=args.train_root,
            dataset=args.dataset,      # <--- 添加这一行，修复 AttributeError
            num_classes=num_classes,
            lpi=args.lpi,
            seed_dataset=args.seed_dataset
        )
        if args.split_protocol == 'standard':
            if args.slice == 1:
                train_split, test_split = ['fold2', 'fold3', 'fold4', 'fold5'], ['fold1']
            elif args.slice == 2:
                train_split, test_split = ['fold1', 'fold3', 'fold4', 'fold5'], ['fold2']
            elif args.slice == 3:
                train_split, test_split = ['fold1', 'fold2', 'fold4', 'fold5'], ['fold3']
            elif args.slice == 4:
                train_split, test_split = ['fold1', 'fold2', 'fold3', 'fold5'], ['fold4']
            elif args.slice == 5:
                train_split, test_split = ['fold1', 'fold2', 'fold3', 'fold4'], ['fold5']
            else:
                raise ValueError(f"Invalid standard slice index: {args.slice}")
        elif args.split_protocol == 'pals_3fold':
            if args.slice == 1:
                train_split, test_split = ['fold1', 'fold4', 'fold5'], ['fold3']
            elif args.slice == 2:
                train_split, test_split = ['fold1', 'fold2', 'fold5'], ['fold4']
            elif args.slice == 3:
                train_split, test_split = ['fold1', 'fold2', 'fold3'], ['fold5']
            else:
                raise ValueError(f"Invalid PALS 3-fold slice index: {args.slice}")
        else:
            raise ValueError(f"Invalid split protocol: {args.split_protocol}")

        base_train_ds = Crowdsource(lpi_args, splits=train_split, transform=None)
        test_ds = Crowdsource(lpi_args, splits=test_split, transform=test_t)

        # 众包数据集使用带有权重的 weights 作为初始分布
        base_train_ds.initial_dist = base_train_ds.weights.copy()
        logger.info(f" 💡 [Crowd Mode] Initialized with weighted candidates (confidence-aware).")

    apply_external_source_if_requested(args, base_train_ds, logger, log_dir)

    # 备份原始噪声标签 (Static Anchor),用于可靠集筛选的基准
    if not hasattr(base_train_ds, 'original_soft_labels'):
        base_train_ds.original_soft_labels = base_train_ds.soft_labels.copy()
        logger.info(" 🔒 [Backup] Original noisy soft labels backed up for robust screening.")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 4. 模型与状态初始化
    encoder, feature_dim = get_base_encoder(args.network, args.dataset)
    encoder = encoder.to(device)
    classifier = nn.Linear(feature_dim, num_classes).to(device)

    # 注意: state_manager 使用完整 schedule horizon, 便于 --max_run_epochs 验证时保持原 e500/e100 动态标尺。
    state_manager = TemporalStateManager(len(base_train_ds), num_classes, schedule_epochs, history_len=args.history_len, use_disambiguation=True)

        # 差异化学习率策略 (Fine-tuning 范式)
    # if args.dataset in ['Treeversity', 'Benthic', 'Plankton']:
    if args.dataset in [ 'Treeversity', 'Benthic', 'Plankton']:
        # 预训练骨干网络使用较小的学习率 (通常为基础 LR 的 0.1 或 0.01)
        encoder_lr = args.lr * 0.01
        logger.info(f"Fine-tuning mode: Encoder LR={encoder_lr}, Classifier LR={args.lr}")
        optimizer = optim.SGD([
            {'params': encoder.parameters(), 'lr': encoder_lr},
            {'params': classifier.parameters(), 'lr': args.lr}
        ], momentum=args.momentum, weight_decay=args.wd)
    else:
        # 从头训练 (CIFAR等) 使用统一学习率
        optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()),
                              lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    # [关键] Phase 1 调度器:T_max = args.epochs
    # 动态解析学习率调度器
    if args.lr_scheduler == 'step':
        # 假设命令行或 args 中存在默认的衰减轮次，例如 [60, 120, 160]
        milestones = getattr(args, 'lr_decay_epochs', [60, 120, 160])
        gamma = getattr(args, 'lr_decay_rate', 0.2)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        logger.info(f"Using StepLR: milestones={milestones}, gamma={gamma}")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        logger.info("Using CosineAnnealingLR")
    best_test_acc = 0.0
    test_acc = 0.0
    start_epoch = 0

    # 初始化特征空间原型 (Prototypes)
    proto_manager = PrototypeManager(num_classes, feature_dim, ema_alpha=0.9, device=device)

    # --- 提前初始化 DataLoader 以避免内核级重建开销 ---
    total_target_num = len(base_train_ds)
    global_labels = torch.full((total_target_num,), -1, dtype=torch.long, device=device)
    global_is_reliable = torch.zeros(total_target_num, dtype=torch.bool, device=device)

    feature_dataset = FeatureExtractionDataset(
        base_train_ds,
        weak_t,
        strong_t if args.feature_extract_view == 'weak_strong_fusion' else None,
    )
    feature_loader = DataLoader(feature_dataset,
                                batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True,
                                persistent_workers=False)

    initial_weights = [1.0] * total_target_num
    dynamic_sampler = DynamicWeightedRandomSampler(initial_weights, total_target_num)
    unified_image_dataset = ImageOnlyDataset(base_train_ds, weak_t, strong_t)
    unified_loader = DataLoader(unified_image_dataset, batch_size=args.batch_size,
                                sampler=dynamic_sampler, num_workers=args.num_workers,
                                pin_memory=True, drop_last=True,
                                persistent_workers=False)

    if resume_checkpoint_path:
        if not os.path.exists(resume_checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
        checkpoint = _safe_torch_load(resume_checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        classifier.load_state_dict(checkpoint['classifier'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        _load_prototype_state_dict(proto_manager, checkpoint.get('prototype_manager'), device)
        _load_temporal_state_dict(state_manager, checkpoint.get('temporal_state_manager'))
        best_test_acc = float(checkpoint.get('best_test_acc', 0.0))
        test_acc = float(checkpoint.get('test_acc', 0.0))
        start_epoch = int(checkpoint.get('next_epoch', checkpoint.get('epoch', 0)))
        _restore_rng_state(checkpoint.get('rng_state'))
        logger.info(
            f"✅ Resumed at next epoch {start_epoch + 1}/{total_epochs} | "
            f"Best={best_test_acc:.2f}% | Last={test_acc:.2f}%"
        )
    elif getattr(args, 'checkpoint_every_epoch', False):
        logger.info("💾 Checkpointing enabled: saving epoch_XXXX.pt, latest.pt, best.pt, epoch_metrics.csv, and epoch_metrics.jsonl.")

    # ==============================================================================
    # 5. 主训练循环(仅第一阶段)
    # ==============================================================================
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        logger.info(f"======== Epoch {epoch+1}/{total_epochs} ========")

        # 5.1 特征提取与可靠性筛选
        features, model_preds = get_features(
            encoder,
            classifier,
            feature_loader,
            device,
            feature_extract_view=args.feature_extract_view,
        )

        class MockTrainloader:
            def __init__(self, dataset): self.dataset = dataset

        # 执行高级筛选
        selected_mask, selected_labels, knn_pl, model_pl, knn_scores = \
            reliable_pseudolabel_selection_advanced(logger, args, device, MockTrainloader(base_train_ds), features, epoch, state_manager, model_preds, proto_manager)

        # 1. 更新原型 (仅使用可靠集)
        proto_manager.update(features, selected_mask.bool(), selected_labels)

        # 2. 全局原型预测 & 历史队列更新 (在重置判断之前执行,以确保获取最新的 Salvage Set)
        _, proto_preds_all = proto_manager.predict(features)

        # 3. 三方队列共识更新(Model / KNN / Proto)+ 稳定打捞集提取
        with torch.no_grad():
            proto_logits = torch.matmul(features, proto_manager.prototypes.T) / 0.1
            proto_soft_preds = F.softmax(proto_logits, dim=1)

        state_manager.update_tri_consensus(
            is_reliable_mask=selected_mask,
            p_model=model_preds,
            p_knn=knn_scores,
            p_proto=proto_soft_preds,
        )
        tri_stable_mask, tri_stable_labels = state_manager.get_stable_tri_mask()

        # ==============================================================================
        # Stage-1 only:不做 SYSTEM RESET / 标签覆写 / 蒸馏更新
        # ==============================================================================

        # 4. 诊断日志(三方版:不再统计 EMA 预测)
        reliable_indices = torch.where(selected_mask > 0)[0]
        unreliable_indices = torch.where(selected_mask == 0)[0]
        unrel_indices_list = unreliable_indices.cpu().tolist()

        log_tri_consensus_diagnostics(
            logger, epoch,
            true_labels=torch.tensor(base_train_ds.clean_labels, device=device),
            unreliable_indices=unrel_indices_list,
            model_pl=model_preds.argmax(dim=1),
            knn_pl=knn_pl,
            proto_pl=proto_preds_all,
            device=device
        )

        # ==============================================================================
        # 5.2 构建 active-set-only 采样器
        # ==============================================================================
        logger.info(" -> [Active-Set-Only] Constructing Sampler & Refinement...")

        # 准备变量:salvage_mask 基于"三方队列共识"稳定掩码
        salvage_mask = tri_stable_mask
        salvaged_labels = tri_stable_labels

        # 旧的对比逻辑 (仅用于日志)
        old_salvage_mask, _, _, _ = state_manager.get_salvage_mask()
        if salvage_mask is not None:
             with torch.no_grad():
                clean_l = torch.tensor(base_train_ds.clean_labels, device=device)
                acc_tri = (salvaged_labels.to(device)[salvage_mask.to(device).bool()] == clean_l[salvage_mask.to(device).bool()]).float().mean().item() * 100 if salvage_mask.sum() > 0 else 0.0
                num_old = old_salvage_mask.sum().item() if old_salvage_mask is not None else 0
                logger.info(f" 🔬 [Salvage Check] New Tri-Stable: {salvage_mask.sum().item()} (Acc: {acc_tri:.2f}%) | Old Multi-Track: {num_old}")

        # 决定是否在训练中使用打捞样本
        # 策略:只要能打捞出来,就视为 Active Sample (在 Phase 2 尤为重要)
        detected_count = salvage_mask.sum().item() if salvage_mask is not None else 0
        use_salvage_for_training = (detected_count > 0)

        salvage_indices = []
        if use_salvage_for_training:
            salvage_indices = torch.where(salvage_mask > 0)[0].cpu().tolist()
            logger.info(f" 🚀 [Active] Promoting {len(salvage_indices)} salvaged samples to training pool.")

        salvage_set = set(salvage_indices)

        # 排除冲突:Reliable Set 中剔除已经是 Salvage 的 (虽然上面做了互斥,这里双重保险)
        real_reliable_indices = [idx for idx in reliable_indices.cpu().tolist() if idx not in salvage_set]

        # --- 统一权重计算 ---
        total_target = len(base_train_ds)
        ordered_indices = salvage_indices + real_reliable_indices
        sampling_weights_aligned = []
        global_labels.fill_(-1)
        global_is_reliable.zero_()

        total_active_count = len(salvage_indices) + len(real_reliable_indices)
        unified_weight = float(max(0.0, total_target / max(total_active_count, 1)))

        # A. 打捞集 (Salvaged)
        for idx in salvage_indices:
            sampling_weights_aligned.append(unified_weight)
            global_labels[idx] = salvaged_labels[idx]
            global_is_reliable[idx] = True

        # B. 可靠集 (Reliable)
        for idx in real_reliable_indices:
            sampling_weights_aligned.append(unified_weight)
            global_labels[idx] = selected_labels[idx]
            global_is_reliable[idx] = True

        non_active_count = total_target - total_active_count
        logger.info(
            f" >> [Sampler] Active: {total_active_count} | Salvaged: {len(salvage_indices)} | "
            f"Reliable: {len(real_reliable_indices)} | NonActiveExcluded: {non_active_count} | "
            f"Unified Weight: {unified_weight:.2f}x"
        )

        avg_sm_weight = 0.0
        if total_active_count == 0:
            logger.warning(" >> [Active-Set-Only] No active samples this epoch; skipping parameter update.")
        else:
            unified_image_dataset.update_index_map(ordered_indices)
            dynamic_sampler.update_weights(sampling_weights_aligned)

            # ==========================================================================
            # 5.3 执行 active-set-only 训练
            # ==========================================================================
            avg_sm_weight = train_unified_single_stream(
                args, encoder, classifier, device, unified_loader, optimizer,
                logger, num_classes, global_labels, global_is_reliable, proto_manager
            )

        scheduler.step()

        # 评估与保存
        test_acc = evaluate(encoder, classifier, test_loader, device)
        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc

        # 本脚本不使用 EMA teacher / EMA 共识(也不维护 EMA 预测均值)

        epoch_duration = time.time() - epoch_start_time
        metrics = {
            'epoch': int(epoch + 1),
            'test_acc': round(float(test_acc), 6),
            'best_test_acc': round(float(best_test_acc), 6),
            'is_best': bool(is_best),
            'lr': float(optimizer.param_groups[0]['lr']),
            'epoch_time_sec': round(float(epoch_duration), 4),
            'avg_sm_weight': round(float(avg_sm_weight), 6),
            'active_count': int(total_active_count),
            'salvaged_count': int(len(salvage_indices)),
            'reliable_count': int(len(real_reliable_indices)),
            'non_active_count': int(non_active_count),
            'unified_weight': round(float(unified_weight), 6),
            'max_w_model': float(getattr(args, 'max_w_model', 0.5)),
            'model_warmup_epochs': int(getattr(args, 'model_warmup_epochs', 10)),
            'seed': int(args.seed),
        }
        _append_epoch_metrics(log_dir, metrics)

        checkpoint_path = None
        if getattr(args, 'checkpoint_every_epoch', False):
            checkpoint_path = _save_epoch_checkpoint(
                log_dir, epoch + 1, args, encoder, classifier, optimizer, scheduler,
                proto_manager, state_manager, best_test_acc, test_acc, metrics, is_best=is_best
            )

        wandb.log({'Test Accuracy': test_acc, 'Best Accuracy': best_test_acc, 'LR': optimizer.param_groups[0]['lr']}, step=epoch+1)
        ckpt_msg = f" | Checkpoint={checkpoint_path}" if checkpoint_path else ""
        logger.info(f"Epoch {epoch+1} Summary: Acc={test_acc:.2f}% | Best={best_test_acc:.2f}% | Time: {epoch_duration:.2f}s{ckpt_msg}\n")

    wandb.finish()
    return best_test_acc, test_acc, time.time() - start_time
# ==============================================================================
#                      MAIN (MODIFIED FOR DUAL STATS)
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()

    # (确保 wandb 已登录)
    # try:
    #     wandb.login()
    # except:
    #     print("Wandb login failed. Set wandb mode to 'disabled'.")
    #     wandb.init(mode="disabled")
    wandb.init(mode="disabled")
    all_best_accuracies, all_final_epoch_accuracies, all_durations = [], [], []

    master_log_dir = os.path.join(args.out, args.exp_name)
    master_logger = setup_logger(master_log_dir, "master_log.txt", is_master=True)
    master_logger.info("========================= Starting Experiment Series =========================")
    master_logger.info(f"Base Settings: {vars(args)}\n" + "="*80)

    for i, seed in enumerate(args.seeds):
        run_args = copy.deepcopy(args)
        run_args.seed = seed

        master_logger.info(f"--- Starting Run {i+1}/{len(args.seeds)} with Seed: {seed} ---")

        # <<< --- MODIFICATION START --- >>>
        best_acc, final_acc, duration = run_single_experiment(run_args)

        all_best_accuracies.append(best_acc)
        all_final_epoch_accuracies.append(final_acc) # 收集 Final Acc
        all_durations.append(duration)

        master_logger.info(f"--- Run {i+1} Finished. Duration: {duration/60.0:.2f} min | Best Acc: {best_acc:.2f}% | Final Acc: {final_acc:.2f}% ---\n")
        # <<< --- MODIFICATION END --- >>>

    # <<< --- MODIFICATION START --- >>>
    mean_best_acc = np.mean(all_best_accuracies)
    std_best_acc = np.std(all_best_accuracies)
    mean_final_acc = np.mean(all_final_epoch_accuracies) # 计算 Final Acc 均值
    std_final_acc = np.std(all_final_epoch_accuracies)   # 计算 Final Acc 标准差
    avg_duration_minutes = np.mean(all_durations) / 60
    # <<< --- MODIFICATION END --- >>>

    master_logger.info("========================= FINAL SUMMARY =========================")
    master_logger.info(f"Experiment Name: {args.exp_name}\n")
    master_logger.info(f"Average Run Duration: {avg_duration_minutes:.2f} min\n")

    # <<< --- MODIFICATION START --- >>>
    master_logger.info(f"Individual Best Accuracies: {[f'{acc:.2f}%' for acc in all_best_accuracies]}")
    master_logger.info(f"Individual Final Epoch Accuracies: {[f'{acc:.2f}%' for acc in all_final_epoch_accuracies]}")

    master_logger.info(f"--> Final Reported (Best Acc): {mean_best_acc:.2f}% ± {std_best_acc:.2f}%")
    master_logger.info(f"--> Final Reported (Final Epoch Acc): {mean_final_acc:.2f}% ± {std_final_acc:.2f}%")
    # <<< --- MODIFICATION END --- >>>

    master_logger.info("="*80)

"""
[Experiment Lineage / 实验溯源]
Base File / 基准文件: 结合投影_众包_自节点_自适应渐进融合.py
Base Commit / 基准提交: N/A (Latest up to 2026-3-13)
Experiment Goal / 实验目的: 动态根据共识率与可靠率关系自适应决定传播深度 (取消固定3次传播)
Modifications / 修改内容:
    - [EN] Added --adaptive_prop_depth and --expected_rel_ratio to dynamically skip Stage 3 if over-sharpening (high consensus, low reliability volume) is detected.
    - [ZH] 新增自适应截断机制。当共识度高（候选集命中率高）但可靠集数量太少时，跳过第三次传播以防类塌陷。
"""

