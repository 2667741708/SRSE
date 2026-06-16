import math

import torch
import torch.nn.functional as F


def _normalize_rows(x, eps=1e-12):
    return x / (x.sum(dim=1, keepdim=True) + eps)


@torch.no_grad()
def _chunked_knn(features, k, chunk_size):
    n = features.shape[0]
    k_eff = min(int(k) + 1, n)
    feats = F.normalize(features.float(), dim=1)
    sims, indices = [], []
    database_t = feats.t()
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        sim = feats[start:end] @ database_t
        vals, idx = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=True)
        sims.append(vals.clamp_min(0.0))
        indices.append(idx)
    return torch.cat(sims, dim=0), torch.cat(indices, dim=0)


@torch.no_grad()
def _topology_daes_weights(raw_d, neighbors, current_soft_labels, args):
    eps = 1e-12
    num_classes = int(args.num_class)
    labels = _normalize_rows(current_soft_labels.float(), eps)

    linear_weights = raw_d / (raw_d.sum(dim=1, keepdim=True) + eps)
    neighbor_labels = labels[neighbors]
    p_knn = (linear_weights.unsqueeze(-1) * neighbor_labels).sum(dim=1)
    p_knn = _normalize_rows(p_knn, eps)

    p_self = labels
    rel_mode = getattr(args, 'nse_topology_rel_mode', 'masked_entropy')
    if rel_mode == 'masked_entropy':
        masked_scores = p_knn * p_self
        masked_prob = _normalize_rows(masked_scores, eps)
        norm_score = -(masked_prob.clamp_min(eps) * masked_prob.clamp_min(eps).log()).sum(dim=1)
        norm_score = norm_score / (math.log(num_classes) + eps)
    elif rel_mode == 'kl':
        kl_self_mode = getattr(args, 'nse_kl_self_mode', 'with_self')
        if kl_self_mode == 'no_self' and neighbors.shape[1] > 1:
            nei_d = raw_d[:, 1:]
            nei_idx = neighbors[:, 1:]
            nei_w = nei_d / (nei_d.sum(dim=1, keepdim=True) + eps)
            nei_labels = labels[nei_idx]
            p_knn_for_kl = (nei_w.unsqueeze(-1) * nei_labels).sum(dim=1)
            p_knn_for_kl = _normalize_rows(p_knn_for_kl, eps)
        else:
            p_knn_for_kl = p_knn
        norm_score = -(p_self * p_knn_for_kl.clamp_min(eps).log()).sum(dim=1)
        norm_score = norm_score / (math.log(num_classes) + eps)
    elif rel_mode == 'agree':
        agree_mass = (p_knn * p_self).sum(dim=1).clamp(min=eps, max=1.0)
        norm_score = -agree_mass.log() / (math.log(num_classes) + eps)
    else:
        raise ValueError('Unknown nse_topology_rel_mode: {}'.format(rel_mode))

    gamma = float(getattr(args, 'nse_topology_rel_gamma', 2.0))
    reliability_scores = torch.exp(-gamma * (norm_score ** 2))

    att_temp = float(getattr(args, 'nse_daes_spatial_temp', 0.5))
    base_tau = float(getattr(args, 'nse_daes_base_tau', 0.1))
    entropy_coeff = float(getattr(
        args, 'nse_daes_entropy_coeff',
        getattr(args, 'nse_entropy_coeff', 0.5),
    ))
    sim_power = float(getattr(args, 'nse_daes_sim_power', 2.0))

    spatial_weights = F.softmax(raw_d / att_temp, dim=1).unsqueeze(-1)
    local_mean = (neighbor_labels * spatial_weights).sum(dim=1)
    local_entropy = -(local_mean.clamp_min(eps) * local_mean.clamp_min(eps).log()).sum(dim=1)
    norm_entropy = local_entropy / (math.log(num_classes) + eps)
    tau_dynamic = (base_tau + (norm_entropy.pow(2) * entropy_coeff)).unsqueeze(1)

    scaled_sim = raw_d.clamp_min(0.0).pow(sim_power) / tau_dynamic
    max_val, _ = scaled_sim.max(dim=1, keepdim=True)
    daes_weights = torch.exp(scaled_sim - max_val.detach())
    neighbor_reliabilities = reliability_scores[neighbors]
    return daes_weights * neighbor_reliabilities


@torch.no_grad()
def _propagate_topology_daes(raw_d, neighbors, source, args):
    weights = _topology_daes_weights(raw_d, neighbors, source, args)
    votes = (weights.unsqueeze(-1) * source[neighbors]).sum(dim=1)
    return F.softmax(votes, dim=1)


@torch.no_grad()
def _filter_by_candidate_quantile(scores, source_prior, args):
    eps = 1e-12
    max_p, max_idx = scores.max(dim=1)
    in_source = source_prior.gather(1, max_idx.view(-1, 1)).view(-1) > 0
    discrepancy = -scores.clamp_min(eps).log()

    is_rel = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    counts = torch.bincount(max_idx[in_source], minlength=int(args.num_class)).float()
    limit = torch.quantile(counts, float(getattr(args, 'nse_delta', 0.25))) if counts.numel() else torch.tensor(0.0, device=scores.device)

    for c in range(int(args.num_class)):
        idx_c = torch.where(in_source & (max_idx == c))[0]
        if idx_c.numel() == 0:
            continue
        k_c = min(int(limit.item()), idx_c.numel())
        if k_c < 1:
            continue
        top = torch.topk(discrepancy[idx_c, c], k=k_c, largest=False).indices
        is_rel[idx_c[top]] = True

    if is_rel.sum() == 0:
        fallback = min(max(1, int(0.01 * scores.shape[0])), scores.shape[0])
        top = torch.topk(max_p * in_source.float(), k=fallback, largest=True).indices
        is_rel[top] = True
    return is_rel, max_p, max_idx, counts, limit


@torch.no_grad()
def nse_reliable_set_selection(args, epoch, sel_stats, train_givenY):
    """Replace only PiCO+'s prototype-distance clean/noisy split.

    The selector mirrors the NSE topology-DAES reliable-set path:
    candidate source -> topology-DAES propagation -> candidate-constrained
    model fusion -> topology-DAES propagation -> per-class quantile selection.
    PiCO+'s losses, EMA confidence target, prototypes, queues, MixUp, and
    unreliable-example branch remain unchanged.
    """
    device = sel_stats['is_rel'].device
    seen = sel_stats.get('seen', torch.ones_like(sel_stats['is_rel'])).bool()
    if seen.sum() < max(2, int(args.nse_k) + 1):
        print('[NSESplit] not enough stored features; keeping all samples reliable')
        sel_stats['is_rel'] = torch.ones_like(sel_stats['is_rel']).bool()
        return

    features = sel_stats['features'].to(device)
    source_prior = train_givenY.float().to(device)
    source_prior = _normalize_rows(source_prior)
    raw_d, neighbors = _chunked_knn(
        features,
        int(getattr(args, 'nse_k', 15)),
        int(getattr(args, 'nse_chunk_size', 1024)),
    )

    stage1 = _propagate_topology_daes(raw_d, neighbors, source_prior, args)

    model_probs = _normalize_rows(sel_stats['model_probs'].to(device))
    if model_probs.sum() > 0:
        warmup = max(float(getattr(args, 'nse_model_warmup_epochs', 10.0)), 1.0)
        max_w_model = float(getattr(args, 'nse_model_weight', 0.5))
        w_model = max_w_model * min(1.0, float(epoch) / warmup)
        p_model_effective = w_model * model_probs + (1.0 - w_model) * stage1.detach()
        conf_knn = stage1.max(dim=1, keepdim=True)[0]
        conf_model = model_probs.max(dim=1, keepdim=True)[0]
        r_i = conf_knn / (conf_knn + conf_model + 1e-12)
        prior_effective = _normalize_rows(p_model_effective * source_prior)
        fused = r_i * stage1 + (1.0 - r_i) * prior_effective
        stage2_input = _normalize_rows(fused)
    else:
        w_model = 0.0
        r_i = torch.ones(stage1.shape[0], 1, device=device)
        stage2_input = stage1

    final_scores = _propagate_topology_daes(raw_d, neighbors, stage2_input, args)
    is_rel, max_p, _max_idx, counts, limit = _filter_by_candidate_quantile(final_scores, source_prior, args)
    sel_stats['is_rel'] = is_rel

    print(
        '[NSESplit:topology_daes] epoch={} reliable={} total={} ratio={:.4f} '
        'limit={} w_model={:.4f} r_i_mean={:.4f} maxp_mean={:.4f} count_q25={:.2f}'.format(
            epoch,
            int(is_rel.sum().item()),
            int(is_rel.numel()),
            float(is_rel.float().mean().item()),
            int(limit.item()) if torch.is_tensor(limit) else int(limit),
            float(w_model),
            float(r_i.mean().item()),
            float(max_p.mean().item()),
            float(torch.quantile(counts, 0.25).item()) if counts.numel() else 0.0,
        )
    )
