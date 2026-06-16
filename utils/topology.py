# -*- coding: utf-8 -*-
"""
Module: utils/topology.py
Decoupled from: v3_2passKNN_refactored.py
Decoupling baseline: commit 6d3088f (refactor/true-decoupled-minimal)
Description: 拓扑亲和度计算、时序状态管理、权重矩阵构建
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
import logging
from collections import deque


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
        )
    elif mode == 'topology_daes':
        refined_D = get_topology_daes_affinity(raw_D, neighbors_indices, ref_soft_labels, args)
    elif mode == 'exp':
        refined_D = torch.exp(raw_D / 0.1)
    else: # 'linear' or 'none'
        refined_D = raw_D
        
    return refined_D


