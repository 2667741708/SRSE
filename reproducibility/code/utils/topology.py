# -*- coding: utf-8 -*-
"""SRSE reproducibility implementation."""
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
        
        
        self.tri_consensus_history = deque(maxlen=history_len)
        
        
        self.is_reliable_history = deque(maxlen=history_len)
        
        
        
        
        self.pruned_pl_history = deque(maxlen=history_len)
        
        
        self.geo_pl_history = deque(maxlen=history_len)
        
        
        self.proto_pl_history = deque(maxlen=history_len)
        
        
        self.prob_ema = torch.ones(num_samples, num_classes) / num_classes
        self.ema_m = 0.995 

    def update_ema(self, current_model_probs):
        """Implement update_ema."""
        self.prob_ema = self.ema_m * self.prob_ema + (1 - self.ema_m) * current_model_probs.cpu()

    def update_history(self, is_reliable_mask, pruned_pl, geo_pl=None, proto_pl=None):
        """Record reliability and prediction histories for consensus recovery."""
        self.is_reliable_history.append(is_reliable_mask.cpu().bool())
        self.pruned_pl_history.append(pruned_pl.cpu().long())
        
        if geo_pl is not None:
            self.geo_pl_history.append(geo_pl.cpu().long())
        else:
            self.geo_pl_history.append(torch.full_like(pruned_pl, -1).cpu().long())

        
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
        """Implement get_salvage_mask."""
        # 1. Base safety check
        curr_len = len(self.is_reliable_history)
        if curr_len == 0:
            return None, None, None, None

        # 2. Stack History
        rel_stack = torch.stack(list(self.is_reliable_history)) 
        pl_stack = torch.stack(list(self.pruned_pl_history))

        # 3. Calculate Metrics
        
        always_unreliable = (rel_stack.sum(dim=0) == 0)

        
        knn_consistency = (pl_stack == pl_stack[-1:]).all(dim=0)

        
        if len(self.geo_pl_history) > 0:
            geo_stack = torch.stack(list(self.geo_pl_history))
            valid_geo = (geo_stack != -1).all(dim=0)
            geo_consistency = (geo_stack == geo_stack[-1:]).all(dim=0) & valid_geo
        else:
            geo_consistency = torch.zeros_like(knn_consistency, dtype=torch.bool)
            geo_stack = pl_stack # Fallback

        
        if len(self.proto_pl_history) > 0:
            proto_stack = torch.stack(list(self.proto_pl_history))
            valid_proto = (proto_stack != -1).all(dim=0)
            proto_consistency = (proto_stack == proto_stack[-1:]).all(dim=0) & valid_proto
        else:
            
            
            proto_consistency = torch.zeros_like(knn_consistency, dtype=torch.bool)
            proto_stack = pl_stack # Fallback

        
        cross_track_agreement = (pl_stack[-1] == geo_stack[-1]) & (geo_stack[-1] == proto_stack[-1])

        
        knn_src_mask = always_unreliable & knn_consistency
        geo_src_mask = always_unreliable & geo_consistency

        # 5. Combine Masks (Strict Intersection)
        
        salvage_mask = knn_src_mask & geo_src_mask & proto_consistency & cross_track_agreement

        # 6. Determine Labels
        
        final_salvaged_labels = self.pruned_pl_history[-1].clone()

        return salvage_mask, final_salvaged_labels, knn_src_mask, geo_src_mask


    def update_tri_consensus(self, is_reliable_mask, p_model, p_knn, p_proto):
        """Implement update_tri_consensus."""
        
        pl_m = p_model.argmax(dim=1).cpu()
        pl_k = p_knn.argmax(dim=1).cpu()
        pl_p = p_proto.argmax(dim=1).cpu()

        tri_mask = (pl_m == pl_k) & (pl_k == pl_p)

        
        snapshot = torch.where(tri_mask, pl_m, torch.tensor(-1))
        self.tri_consensus_history.append(snapshot)


    def get_stable_tri_mask(self):
        """Implement get_stable_tri_mask."""
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
    """Build topology-guided affinity weights for label propagation."""
    N, K_plus_1 = neighbors_indices.shape

    
    linear_weights = raw_D 
    linear_weights = linear_weights / (linear_weights.sum(dim=1, keepdim=True) + eps)

    neighbor_labels = F.embedding(neighbors_indices, current_soft_labels)  
    knn_scores_smooth = (linear_weights.unsqueeze(-1) * neighbor_labels).sum(dim=1)
    p_knn = knn_scores_smooth / (knn_scores_smooth.sum(dim=1, keepdim=True) + eps)

    
    p_self = current_soft_labels / (current_soft_labels.sum(dim=1, keepdim=True) + eps)
    if rel_mode == 'masked_entropy':
        masked_scores = p_knn * p_self
        masked_prob = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + eps)
        norm_score = -torch.sum(masked_prob * torch.log(masked_prob + eps), dim=1) / (np.log(num_classes) + eps)
    elif rel_mode == 'kl':
        # ===========================================================================
        
        
        
        
        
        
        # ===========================================================================
        if kl_self_mode == 'no_self' and K_plus_1 > 1:
            
            # Exclude self-node (index 0), use pure neighbors [:, 1:] only
            nei_D = raw_D[:, 1:]        # [N, K]
            nei_idx = neighbors_indices[:, 1:]  # [N, K]
            nei_w = nei_D / (nei_D.sum(dim=1, keepdim=True) + eps)
            nei_labels = F.embedding(nei_idx, current_soft_labels)
            nei_agg = (nei_w.unsqueeze(-1) * nei_labels).sum(dim=1)
            p_knn_for_kl = nei_agg / (nei_agg.sum(dim=1, keepdim=True) + eps)
        else:
            
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

    
    # Final affinity matrix always uses the full [N, K+1] matrix.
    reliability_scores_expanded = reliability_scores.unsqueeze(1) 
    all_reliabilities = F.embedding(neighbors_indices, reliability_scores_expanded).squeeze(-1)
    refined_sim = raw_D * all_reliabilities

    return refined_sim, reliability_scores


def get_topology_daes_affinity(raw_D, neighbors_indices, current_soft_labels, args):
    """Build the combined Topology-DAES affinity matrix."""
    eps = getattr(args, 'topology_rel_eps', 1e-12)
    N, K_plus_1 = neighbors_indices.shape
    num_classes = args.num_classes

    
    # Stage 1: Topology (always compute p_knn from full matrix)
    linear_weights = raw_D / (raw_D.sum(dim=1, keepdim=True) + eps) 
    neighbor_labels_for_top = F.embedding(neighbors_indices, current_soft_labels) 
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
        
        
        
        
        # Modified: no_self mode aggregates only pure neighbors for KL reliability scoring.
        #           DAES neighborhood entropy computation is unaffected.
        # ===========================================================================
        kl_self_mode = getattr(args, 'kl_self_mode', 'with_self')
        if kl_self_mode == 'no_self' and K_plus_1 > 1:
            
            # Exclude self-node; compute KL reliability from pure neighbors only
            nei_D = raw_D[:, 1:]        # [N, K]
            nei_idx = neighbors_indices[:, 1:]  # [N, K]
            nei_w = nei_D / (nei_D.sum(dim=1, keepdim=True) + eps)
            nei_labels = F.embedding(nei_idx, current_soft_labels)
            nei_agg = (nei_w.unsqueeze(-1) * nei_labels).sum(dim=1)
            p_knn_for_kl = nei_agg / (nei_agg.sum(dim=1, keepdim=True) + eps)
        else:
            
            # with_self: original behavior, p_knn includes self-node
            p_knn_for_kl = p_knn
        norm_score = -torch.sum(p_self * torch.log(p_knn_for_kl + eps), dim=1) / (np.log(num_classes) + eps)
    elif rel_mode == 'agree':
        agree_mass = torch.sum(p_knn * p_self, dim=1).clamp(min=eps, max=1.0)
        norm_score = (-torch.log(agree_mass)) / (np.log(num_classes) + eps)

    gamma = float(getattr(args, 'topology_rel_gamma', 2.0))
    reliability_scores = torch.exp(-gamma * (norm_score ** 2))

    
    # Stage 2: DAES (always uses full matrix, unaffected by kl_self_mode)
    att_temp = getattr(args, 'daes_spatial_temp', 0.5)
    base_tau = getattr(args, 'daes_base_tau', 0.1)
    entropy_coeff = getattr(args, 'daes_entropy_coeff', 0.5)
    sim_power = getattr(args, 'daes_sim_power', 2.0)
    
    
    spatial_weights = F.softmax(raw_D / att_temp, dim=1).unsqueeze(-1)
    neighbor_labels = F.embedding(neighbors_indices, current_soft_labels)
    
    local_mean = (neighbor_labels * spatial_weights).sum(dim=1)
    local_entropy = -torch.sum(local_mean * torch.log(local_mean + eps), dim=1)
    norm_entropy = local_entropy / np.log(num_classes)
    
    tau_dynamic = (base_tau + (torch.pow(norm_entropy, 2) * entropy_coeff)).unsqueeze(1)

    
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
    """Implement get_weight_matrix."""
    if mode == 'daes':
        refined_D = get_adaptive_affinity_matrix(raw_D, neighbors_indices, ref_soft_labels, args)
    elif mode == 'topology':
        
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


