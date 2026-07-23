# -*- coding: utf-8 -*-
"""
Module: utils/engine.py
Decoupled from: v3_2passKNN_refactored.py
Decoupling baseline: commit 6d3088f (refactor/true-decoupled-minimal)
Description: 评估引擎、特征提取、KNN搜索、混合目标构建、诊断日志
"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np


def build_reliable_mixup_targets(rel_labels, knn_scores_batch, num_classes, args, device, logger=None, log_once=False):
    s_labels = F.one_hot(rel_labels.long(), num_classes).float()
    dataset_name = getattr(args, 'dataset', '')

    # Crowd datasets keep the one-hot target, while CIFAR uses a
    # KNN-residual target when neighborhood scores are available.
    if dataset_name in {'Benthic', 'Treeversity', 'Plankton'}:
        target_labels = s_labels
        strategy = 'crowd_zero_smoothing'
    elif knn_scores_batch is not None:
        batch_knn = knn_scores_batch.to(device).float()
        knn_dist = batch_knn / (batch_knn.sum(dim=1, keepdim=True) + 1e-8)
        max_prob = knn_dist.max(dim=1, keepdim=True)[0]
        alpha = 0.3 * (1.0 - max_prob)
        target_labels = (1.0 - alpha) * s_labels + alpha * knn_dist
        strategy = f'knn_residual_auto(mean_alpha={alpha.mean().item():.4f})'
    else:
        target_labels = s_labels
        strategy = 'one_hot_fallback'

    if logger is not None and log_once:
        logger.info(f"  [Target Builder] dataset={dataset_name} strategy={strategy}")
    return target_labels



def get_features(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval(); all_features, all_predictions, all_indices = [], [], []
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        with autocast():
            features = encoder(images)
            predictions = F.softmax(classifier(features), dim=1)
        all_features.append(F.normalize(features.float())); all_predictions.append(predictions.float()); all_indices.append(indices.cpu())
    all_features, all_predictions, all_indices = torch.cat(all_features), torch.cat(all_predictions), torch.cat(all_indices)
    return all_features[torch.argsort(all_indices)], all_predictions[torch.argsort(all_indices)]


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


def log_tri_consensus_diagnostics(logger, epoch, true_labels, 
                                  unreliable_indices, 
                                  model_pl, knn_pl, proto_pl,
                                  device):
    """诊断：Model vs KNN vs Prototype 在不可靠集上的多方博弈（三方版）。

    说明：该脚本不再把 EMA 预测作为共识参与方，因此这里也不再统计 EMA。
    """
    if len(unreliable_indices) == 0:
        return

    logger.info(f"\n📐 [Triangle Diagnostics] Epoch {epoch} | Unreliable Set: {len(unreliable_indices)}")

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

