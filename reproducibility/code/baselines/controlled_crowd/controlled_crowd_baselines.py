#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled crowd baselines for the SRSE crowd table.

The script ports the method-side update rules of label-noise / partial-label
baselines onto one fixed crowd protocol: same fold split, same LPI sampling,
same ImageNet-pretrained R50, and same LR schedule.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    size: int
    mean: tuple
    std: tuple
    grayscale: bool = False


DATASET_SPECS = {
    "Benthic": DatasetSpec(
        num_classes=8,
        size=112,
        mean=(0.34728872821176615, 0.40013687864974884, 0.4110478166769647),
        std=(0.1286915489786319, 0.13644626747739305, 0.14258506692263767),
        grayscale=False,
    ),
    "Plankton": DatasetSpec(
        num_classes=10,
        size=96,
        mean=(0.9663359216202008, 0.9663359216202008, 0.9663359216202008),
        std=(0.10069729102981237, 0.10069729102981237, 0.10069729102981237),
        grayscale=True,
    ),
}


def split_names(slice_id, protocol="standard"):
    if protocol == "standard":
        folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]
        if slice_id < 1 or slice_id > 5:
            raise ValueError(f"standard slice must be in 1..5, got {slice_id}")
        test = [folds[slice_id - 1]]
        train = [f for f in folds if f not in test]
        return train, test

    if protocol == "pals_3fold":
        # Matches the 3-train/1-test crowd split convention in the PALS crowd
        # loader family, e.g. data/crowdsource_soft.py.
        mapping = {
            1: (["fold1", "fold4", "fold5"], ["fold3"]),
            2: (["fold1", "fold2", "fold5"], ["fold4"]),
            3: (["fold1", "fold2", "fold3"], ["fold5"]),
        }
        if slice_id not in mapping:
            raise ValueError(f"pals_3fold slice must be in 1..3, got {slice_id}")
        return mapping[slice_id]

    raise ValueError(f"unknown split protocol: {protocol}")


def sort_labels(labels):
    try:
        return sorted(labels, key=lambda x: int(x))
    except Exception:
        return sorted(labels)


class CrowdPseudoLabelDataset(Dataset):
    def __init__(
        self,
        dataset,
        root,
        num_classes,
        lpi,
        seed_dataset,
        splits,
        transform_weak=None,
        transform_strong=None,
        train=True,
    ):
        self.dataset = dataset
        self.root = Path(root).expanduser()
        self.num_classes = num_classes
        self.lpi = lpi
        self.seed_dataset = seed_dataset
        self.splits = list(splits)
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.train = train

        annotation_file = self.root / "annotations.json"
        if not annotation_file.exists():
            alt = self.root / dataset / "annotations.json"
            if alt.exists():
                self.root = self.root / dataset
                annotation_file = alt
        if not annotation_file.exists():
            raise FileNotFoundError(f"annotations.json not found under {self.root}")

        with annotation_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        entries = raw[0]["annotations"] if isinstance(raw, list) else raw["annotations"]

        img_names = sorted({e["image_path"] for e in entries if e.get("class_label") is not None})
        label_names = sort_labels({e["class_label"] for e in entries if e.get("class_label") is not None})
        label_to_idx = {label: i for i, label in enumerate(label_names)}
        if len(label_names) != num_classes:
            print(
                f"[WARN] {dataset}: num_classes={num_classes}, "
                f"annotations expose {len(label_names)} labels",
                flush=True,
            )

        name_to_idx = {name: i for i, name in enumerate(img_names)}
        folds = {}
        for name in img_names:
            parts = name.replace("\\", "/").split("/")
            if len(parts) < 2:
                raise ValueError(f"Cannot infer fold from image path: {name}")
            folds.setdefault(parts[1], []).append(name_to_idx[name])

        vote_counts = np.zeros((len(img_names), num_classes), dtype=np.float64)
        for entry in entries:
            label = entry.get("class_label")
            if label is None or label not in label_to_idx:
                continue
            img_idx = name_to_idx[entry["image_path"]]
            label_idx = label_to_idx[label]
            if label_idx < num_classes:
                vote_counts[img_idx, label_idx] += 1.0

        sampled = np.zeros_like(vote_counts)
        rng = np.random.default_rng(seed_dataset)
        for i in range(len(img_names)):
            total = vote_counts[i].sum()
            if total <= 0:
                continue
            probs = vote_counts[i] / total
            annots = rng.choice(num_classes, p=probs, size=lpi)
            for a in annots:
                sampled[i, a] += 1.0

        req_ids = []
        for split in self.splits:
            if split not in folds:
                raise ValueError(f"split {split} not found in annotations; found {sorted(folds)}")
            req_ids.extend(folds[split])

        self.data = []
        for i in req_ids:
            rel = Path(img_names[i])
            full = self.root / rel
            if not full.exists():
                alt = self.root.parent / rel
                full = alt if alt.exists() else full
            self.data.append(str(full))

        partial = np.zeros_like(sampled)
        partial[sampled.nonzero()] = 1.0
        weights = sampled / (sampled.sum(axis=1, keepdims=True) + 1e-8)
        clean = vote_counts.argmax(axis=1).astype(np.int64)
        hard = weights.argmax(axis=1).astype(np.int64)

        self.soft_labels = partial[req_ids].astype(np.float32)
        self.weights = weights[req_ids].astype(np.float32)
        self.clean_labels = clean[req_ids].astype(np.int64)
        self.hard_labels = hard[req_ids].astype(np.int64)

        clean_in_candidate = self.soft_labels[np.arange(len(self.clean_labels)), self.clean_labels].sum()
        clean_majority = (self.hard_labels == self.clean_labels).sum()
        print(
            f"[DATA] {dataset} splits={self.splits} n={len(self.data)} "
            f"avg_candidate={self.soft_labels.sum(axis=1).mean():.3f} "
            f"clean_in_candidate={int(clean_in_candidate)}/{len(self.data)} "
            f"clean_majority={int(clean_majority)}/{len(self.data)}",
            flush=True,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            img = Image.open(self.data[index]).convert("RGB")
        except Exception as exc:
            print(f"[WARN] failed to read {self.data[index]}: {exc}", flush=True)
            img = Image.new("RGB", (224, 224))
        if self.train:
            weak = self.transform_weak(img) if self.transform_weak else img
            strong = self.transform_strong(img) if self.transform_strong else weak
            return weak, strong, int(self.hard_labels[index]), index
        x = self.transform_weak(img) if self.transform_weak else img
        return x, int(self.clean_labels[index])


class Identity(nn.Module):
    def forward(self, x):
        return x


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Match the PALS/SRSE crowd protocol: torchvision ImageNet-pretrained R50
        # followed by a 512-d bottleneck and a task classifier.
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 512)
        self.encoder = model
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        feat = self.encoder(x)
        logits = self.fc(feat)
        if return_features:
            return logits, F.normalize(feat, dim=1)
        return logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_transforms(args):
    spec = DATASET_SPECS[args.dataset]
    common = []
    if spec.grayscale:
        common.append(transforms.Grayscale(num_output_channels=3))
    common.extend([transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)])

    weak_ops = [transforms.RandomHorizontalFlip(), transforms.Resize((spec.size, spec.size))]
    strong_ops = [transforms.RandomHorizontalFlip(), transforms.Resize((spec.size, spec.size))]
    if args.strong_aug:
        try:
            strong_ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        except Exception:
            pass
    test_ops = [transforms.Resize((spec.size, spec.size))]
    return (
        transforms.Compose(weak_ops + common),
        transforms.Compose(strong_ops + common),
        transforms.Compose(test_ops + common),
    )


def make_datasets(args):
    spec = DATASET_SPECS[args.dataset]
    if args.num_classes is None:
        args.num_classes = spec.num_classes
    train_splits, test_splits = split_names(args.slice, args.split_protocol)
    transform_w, transform_s, transform_t = make_transforms(args)
    train_set = CrowdPseudoLabelDataset(
        args.dataset,
        args.train_root,
        args.num_classes,
        args.lpi,
        args.seed_dataset,
        train_splits,
        transform_weak=transform_w,
        transform_strong=transform_s,
        train=True,
    )
    eval_train_set = CrowdPseudoLabelDataset(
        args.dataset,
        args.train_root,
        args.num_classes,
        args.lpi,
        args.seed_dataset,
        train_splits,
        transform_weak=transform_t,
        transform_strong=transform_t,
        train=True,
    )
    test_set = CrowdPseudoLabelDataset(
        args.dataset,
        args.train_root,
        args.num_classes,
        args.lpi,
        args.seed_dataset,
        test_splits,
        transform_weak=transform_t,
        train=False,
    )
    return train_set, eval_train_set, test_set


def make_loaders(args):
    train_set, eval_train_set, test_set = make_datasets(args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    eval_train_loader = DataLoader(
        eval_train_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_set, train_loader, eval_train_loader, test_loader


def make_model(args, device):
    model = PretrainedResNet(args.num_classes).to(device)
    return model


def make_optimizer(args, model):
    return torch.optim.SGD(
        [
            {"params": model.encoder.parameters(), "lr": args.lr / 100.0},
            {"params": model.fc.parameters(), "lr": args.lr},
        ],
        momentum=args.momentum,
        weight_decay=args.wd,
    )


def make_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_decay_epochs,
        gamma=args.lr_decay_rate,
    )


def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def one_hot(labels, num_classes):
    return F.one_hot(labels.long(), num_classes=num_classes).float()


def current_lrs(optimizer):
    return ",".join(f"{g['lr']:.6g}" for g in optimizer.param_groups)


@torch.no_grad()
def evaluate(models_to_eval, test_loader, device):
    if not isinstance(models_to_eval, (list, tuple)):
        models_to_eval = [models_to_eval]
    for model in models_to_eval:
        model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = None
        for model in models_to_eval:
            out = model(x)
            logits = out if logits is None else logits + out
        logits = logits / float(len(models_to_eval))
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def train_elr_plus(args, model, optimizer, loader, pred_hist, epoch, device):
    model.train()
    total = 0
    loss_sum = 0.0
    ce_sum = 0.0
    reg_sum = 0.0
    for weak, _strong, hard, indices in loader:
        weak = weak.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        indices = indices.long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(weak)
        probs = F.softmax(logits, dim=1).clamp(min=1e-6, max=1.0 - 1e-6)
        ce = F.cross_entropy(logits, hard)
        with torch.no_grad():
            old = pred_hist[indices].to(device)
            new = args.elr_beta * old + (1.0 - args.elr_beta) * probs
            new = new / new.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pred_hist[indices] = new.detach().cpu()
        q = pred_hist[indices].to(device)
        agreement = (q * probs).sum(dim=1).clamp(max=1.0 - 1e-6)
        reg = torch.log(1.0 - agreement + 1e-6).mean()
        loss = ce + args.elr_lambda * reg
        loss.backward()
        optimizer.step()
        bs = hard.numel()
        total += bs
        loss_sum += loss.item() * bs
        ce_sum += ce.item() * bs
        reg_sum += reg.item() * bs
    return {
        "loss": loss_sum / max(total, 1),
        "ce": ce_sum / max(total, 1),
        "elr_reg": reg_sum / max(total, 1),
    }


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


class ELRPlusFullLoss(nn.Module):
    def __init__(self, num_examples, num_classes, beta, lambda_, coef_step, device):
        super().__init__()
        self.pred_hist = torch.zeros(num_examples, num_classes, device=device)
        self.q = None
        self.beta = beta
        self.lambda_ = lambda_
        self.coef_step = coef_step
        self.num_classes = num_classes

    @torch.no_grad()
    def update_hist(self, out, indices, mix_index, mixup_l):
        y_pred = F.softmax(out, dim=1)
        y_pred = y_pred / y_pred.sum(dim=1, keepdim=True).clamp_min(1e-12)
        indices = indices.to(self.pred_hist.device).long()
        self.pred_hist[indices] = self.beta * self.pred_hist[indices] + (1.0 - self.beta) * y_pred
        self.q = mixup_l * self.pred_hist[indices] + (1.0 - mixup_l) * self.pred_hist[indices][mix_index]

    def forward(self, iteration, output, y_labeled):
        y_pred = F.softmax(output, dim=1).clamp(min=1e-4, max=1.0 - 1e-4)
        ce_loss = torch.mean(-torch.sum(y_labeled * F.log_softmax(output, dim=1), dim=-1))
        if self.q is None:
            reg = torch.zeros((), device=output.device)
        else:
            reg = torch.log(1.0 - (self.q.detach() * y_pred).sum(dim=1).clamp(max=1.0 - 1e-6)).mean()
        loss = ce_loss + sigmoid_rampup(iteration, self.coef_step) * (self.lambda_ * reg)
        return loss, ce_loss.detach(), reg.detach()


def mixup_data(x, y, alpha, device):
    if alpha <= 0:
        return x, y, 1.0, torch.arange(x.size(0), device=device)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    mix_index = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1.0 - lam) * x[mix_index]
    mixed_y = lam * y + (1.0 - lam) * y[mix_index]
    return mixed_x, mixed_y, float(lam), mix_index


def update_ema_variables(model, model_ema, global_step, ema_alpha, ema_step, ema_update=True):
    if ema_update:
        alpha = sigmoid_rampup(global_step + 1, ema_step) * ema_alpha
    else:
        alpha = min(1.0 - 1.0 / float(global_step + 1), ema_alpha)
    with torch.no_grad():
        for ema_param, param in zip(model_ema.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)
        for ema_buffer, buffer in zip(model_ema.buffers(), model.buffers()):
            if torch.is_floating_point(ema_buffer):
                ema_buffer.data.mul_(alpha).add_(buffer.data, alpha=1.0 - alpha)
            else:
                ema_buffer.data.copy_(buffer.data)


def train_elr_plus_full_epoch(
    args,
    model,
    model_ema,
    peer_model_ema,
    optimizer,
    scheduler,
    loader,
    criterion,
    epoch,
    global_step,
    device,
):
    model.train()
    model_ema.train()
    peer_model_ema.train()
    total = 0
    local_step = 0
    loss_sum = 0.0
    ce_sum = 0.0
    reg_sum = 0.0
    for weak, _strong, hard, indices in loader:
        weak = weak.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        targets = one_hot(hard, args.num_classes)
        mixed_x, mixed_y, mixup_l, mix_index = mixup_data(weak, targets, args.elr_mixup_alpha, device)

        with torch.no_grad():
            peer_output = peer_model_ema(weak)
            criterion.update_hist(peer_output, indices, mix_index, mixup_l)

        optimizer.zero_grad(set_to_none=True)
        output = model(mixed_x)
        local_step += 1
        loss, ce_loss, reg = criterion(global_step + local_step, output, mixed_y)
        loss.backward()
        optimizer.step()
        update_ema_variables(
            model,
            model_ema,
            global_step + local_step,
            args.elr_ema_alpha,
            args.elr_ema_step,
            args.elr_ema_update,
        )

        bs = hard.numel()
        total += bs
        loss_sum += loss.item() * bs
        ce_sum += ce_loss.item() * bs
        reg_sum += reg.item() * bs

    scheduler.step()
    return {
        "loss": loss_sum / max(total, 1),
        "ce": ce_sum / max(total, 1),
        "elr_reg": reg_sum / max(total, 1),
        "local_step": local_step,
    }


def alim_conf_momentum(args, epoch):
    if args.epochs <= 1:
        return args.alim_conf_ema_end
    ratio = (epoch - 1) / float(args.epochs - 1)
    return args.alim_conf_ema_start + ratio * (args.alim_conf_ema_end - args.alim_conf_ema_start)


def train_alim_onehot(args, model, optimizer, loader, confidence, partial_mask, epoch, device):
    model.train()
    total = 0
    loss_sum = 0.0
    m = alim_conf_momentum(args, epoch)
    for weak, _strong, _hard, indices in loader:
        weak = weak.to(device, non_blocking=True)
        indices = indices.long()
        conf = confidence[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(weak)
        loss = soft_cross_entropy(logits, conf)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            pmask = partial_mask[indices].to(device)
            prior = pmask + args.alim_prior * (1.0 - pmask)
            pred = (probs * prior).argmax(dim=1)
            pseudo = one_hot(pred, args.num_classes)
            updated = m * conf + (1.0 - m) * pseudo
            updated = updated / updated.sum(dim=1, keepdim=True).clamp_min(1e-12)
            confidence[indices] = updated.cpu()

        bs = weak.size(0)
        total += bs
        loss_sum += loss.item() * bs
    return {"loss": loss_sum / max(total, 1), "conf_m": m}


@torch.no_grad()
def per_sample_losses(model, loader, device):
    model.eval()
    n = len(loader.dataset)
    losses = torch.zeros(n, dtype=torch.float32)
    for weak, _strong, hard, indices in loader:
        weak = weak.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        logits = model(weak)
        batch_losses = F.cross_entropy(logits, hard, reduction="none").detach().cpu()
        losses[indices.long()] = batch_losses
    return losses


def clean_prob_from_losses(losses, seed):
    x = losses.detach().cpu().numpy().astype(np.float64)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    try:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, random_state=seed)
        prob = gmm.fit_predict(x.reshape(-1, 1))
        posterior = gmm.predict_proba(x.reshape(-1, 1))
        clean_component = int(np.argmin(gmm.means_.reshape(-1)))
        return torch.from_numpy(posterior[:, clean_component].astype(np.float32))
    except Exception as exc:
        print(f"[WARN] sklearn GMM unavailable ({exc}); using two-means fallback", flush=True)
        c1, c2 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
        for _ in range(20):
            d1 = np.abs(x - c1)
            d2 = np.abs(x - c2)
            assign = d2 < d1
            if (~assign).any():
                c1 = float(x[~assign].mean())
            if assign.any():
                c2 = float(x[assign].mean())
        clean_center, noisy_center = (c1, c2) if c1 < c2 else (c2, c1)
        midpoint = 0.5 * (clean_center + noisy_center)
        temp = max(abs(noisy_center - clean_center) / 6.0, 1e-3)
        prob = 1.0 / (1.0 + np.exp((x - midpoint) / temp))
        return torch.from_numpy(prob.astype(np.float32))


def train_dividemix_warmup(args, model, optimizer, loader, device):
    model.train()
    total = 0
    loss_sum = 0.0
    for weak, _strong, hard, _indices in loader:
        weak = weak.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(weak)
        loss = F.cross_entropy(logits, hard)
        loss.backward()
        optimizer.step()
        bs = hard.numel()
        total += bs
        loss_sum += loss.item() * bs
    return loss_sum / max(total, 1)


def sharpen(targets, temp):
    out = targets.pow(1.0 / temp)
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)


def train_dividemix_model(args, model, peer_model, optimizer, loader, peer_clean_prob, epoch, device):
    model.train()
    peer_model.eval()
    total = 0
    loss_sum = 0.0
    clean_sum = 0.0
    for weak, strong, hard, indices in loader:
        weak = weak.to(device, non_blocking=True)
        strong = strong.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        indices = indices.long()
        with torch.no_grad():
            peer_logits = peer_model(weak)
            peer_probs = F.softmax(peer_logits, dim=1)
            hard_oh = one_hot(hard, args.num_classes)
            clean_w = peer_clean_prob[indices].to(device).unsqueeze(1)
            targets = clean_w * hard_oh + (1.0 - clean_w) * peer_probs
            targets = sharpen(targets, args.dm_temp)

        inputs = torch.cat([weak, strong], dim=0)
        targets_all = torch.cat([targets, targets], dim=0)
        lam = np.random.beta(args.dm_alpha, args.dm_alpha)
        lam = max(lam, 1.0 - lam)
        perm = torch.randperm(inputs.size(0), device=device)
        mixed_x = lam * inputs + (1.0 - lam) * inputs[perm]
        mixed_y = lam * targets_all + (1.0 - lam) * targets_all[perm]

        optimizer.zero_grad(set_to_none=True)
        logits = model(mixed_x)
        loss = soft_cross_entropy(logits, mixed_y)
        loss.backward()
        optimizer.step()

        bs = hard.numel()
        total += bs
        loss_sum += loss.item() * bs
        clean_sum += clean_w.mean().item() * bs
    return {"loss": loss_sum / max(total, 1), "peer_clean_prob": clean_sum / max(total, 1)}


def write_run_header(args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    (out_dir / "args.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_single_model(args, device, train_set, train_loader, eval_train_loader, test_loader, out_dir):
    model = make_model(args, device)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)
    best_acc = -1.0
    best_epoch = 0

    if args.method == "elr_plus":
        pred_hist = torch.zeros(len(train_set), args.num_classes, dtype=torch.float32)
        train_epoch = lambda epoch: train_elr_plus(args, model, optimizer, train_loader, pred_hist, epoch, device)
    elif args.method == "alim_onehot":
        hard = torch.from_numpy(train_set.hard_labels)
        confidence = one_hot(hard, args.num_classes).cpu()
        partial_mask = torch.from_numpy(train_set.soft_labels).float()
        train_epoch = lambda epoch: train_alim_onehot(
            args, model, optimizer, train_loader, confidence, partial_mask, epoch, device
        )
    else:
        raise ValueError(f"single-model runner does not support {args.method}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        metrics = train_epoch(epoch)
        scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{args.epochs} method={args.method} "
            f"train={metrics} test_loss={test_loss:.6f} "
            f"Test Acc: {test_acc:.4f} Best Acc: {best_acc:.4f} "
            f"Best Epoch: {best_epoch} lr={current_lrs(optimizer)} time={elapsed:.2f}s",
            flush=True,
        )
    return {"best_acc": best_acc, "best_epoch": best_epoch, "final_acc": test_acc}


def run_dividemix(args, device, train_set, train_loader, eval_train_loader, test_loader, out_dir):
    model1 = make_model(args, device)
    model2 = make_model(args, device)
    opt1 = make_optimizer(args, model1)
    opt2 = make_optimizer(args, model2)
    sch1 = make_scheduler(args, opt1)
    sch2 = make_scheduler(args, opt2)
    best_acc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        if epoch <= args.dm_warmup_epochs:
            loss1 = train_dividemix_warmup(args, model1, opt1, train_loader, device)
            loss2 = train_dividemix_warmup(args, model2, opt2, train_loader, device)
            metrics = {"warmup_loss1": loss1, "warmup_loss2": loss2}
        else:
            losses1 = per_sample_losses(model1, eval_train_loader, device)
            losses2 = per_sample_losses(model2, eval_train_loader, device)
            prob1 = clean_prob_from_losses(losses1, args.seed + epoch * 2)
            prob2 = clean_prob_from_losses(losses2, args.seed + epoch * 2 + 1)
            metrics1 = train_dividemix_model(args, model1, model2, opt1, train_loader, prob2, epoch, device)
            metrics2 = train_dividemix_model(args, model2, model1, opt2, train_loader, prob1, epoch, device)
            metrics = {"m1": metrics1, "m2": metrics2}
        sch1.step()
        sch2.step()
        test_loss, test_acc = evaluate([model1, model2], test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{args.epochs} method={args.method} "
            f"train={metrics} test_loss={test_loss:.6f} "
            f"Test Acc: {test_acc:.4f} Best Acc: {best_acc:.4f} "
            f"Best Epoch: {best_epoch} lr={current_lrs(opt1)} time={elapsed:.2f}s",
            flush=True,
        )
    return {"best_acc": best_acc, "best_epoch": best_epoch, "final_acc": test_acc}


def run_elr_plus_full(args, device, train_set, train_loader, eval_train_loader, test_loader, out_dir):
    # Official ELR+ trainer-level mechanics: two online networks, two EMA
    # networks, mixup targets, and pred_hist/q updated from the peer EMA output.
    model1 = make_model(args, device)
    model2 = make_model(args, device)
    model_ema1 = copy.deepcopy(model1).to(device)
    model_ema2 = copy.deepcopy(model2).to(device)
    for ema_model in (model_ema1, model_ema2):
        for param in ema_model.parameters():
            param.detach_()

    optimizer1 = make_optimizer(args, model1)
    optimizer2 = make_optimizer(args, model2)
    scheduler1 = make_scheduler(args, optimizer1)
    scheduler2 = make_scheduler(args, optimizer2)
    criterion1 = ELRPlusFullLoss(
        len(train_set),
        args.num_classes,
        beta=args.elr_beta,
        lambda_=args.elr_lambda,
        coef_step=args.elr_coef_step,
        device=device,
    )
    criterion2 = ELRPlusFullLoss(
        len(train_set),
        args.num_classes,
        beta=args.elr_beta,
        lambda_=args.elr_lambda,
        coef_step=args.elr_coef_step,
        device=device,
    )
    train_loader2 = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    best_acc = -1.0
    best_epoch = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        if epoch <= args.elr_warmup_epochs:
            m1 = {"warmup_ce": train_dividemix_warmup(args, model1, optimizer1, train_loader, device), "local_step": len(train_loader)}
            m2 = {"warmup_ce": train_dividemix_warmup(args, model2, optimizer2, train_loader2, device), "local_step": len(train_loader2)}
            scheduler1.step()
            scheduler2.step()
            update_ema_variables(model1, model_ema1, global_step + m1["local_step"], args.elr_ema_alpha, args.elr_ema_step, args.elr_ema_update)
            update_ema_variables(model2, model_ema2, global_step + m2["local_step"], args.elr_ema_alpha, args.elr_ema_step, args.elr_ema_update)
        else:
            m1 = train_elr_plus_full_epoch(
                args,
                model1,
                model_ema1,
                model_ema2,
                optimizer1,
                scheduler1,
                train_loader,
                criterion1,
                epoch,
                global_step,
                device,
            )
            m2 = train_elr_plus_full_epoch(
                args,
                model2,
                model_ema2,
                model_ema1,
                optimizer2,
                scheduler2,
                train_loader2,
                criterion2,
                epoch,
                global_step,
                device,
            )
        global_step += int(m1["local_step"])

        test_loss, test_acc = evaluate([model1, model2], test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{args.epochs} method={args.method} "
            f"train={{'m1': {m1}, 'm2': {m2}}} test_loss={test_loss:.6f} "
            f"Test Acc: {test_acc:.4f} Best Acc: {best_acc:.4f} "
            f"Best Epoch: {best_epoch} lr={current_lrs(optimizer1)} time={elapsed:.2f}s",
            flush=True,
        )
    return {"best_acc": best_acc, "best_epoch": best_epoch, "final_acc": test_acc}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["dividemix", "elr_plus", "elr_plus_full", "alim_onehot"])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--train_root", required=True)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--lpi", type=int, default=3)
    parser.add_argument("--slice", type=int, default=2)
    parser.add_argument("--split_protocol", default="standard", choices=["standard", "pals_3fold"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed_dataset", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_decay_epochs", type=int, nargs="+", default=[60, 80])
    parser.add_argument("--lr_decay_rate", type=float, default=0.2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--strong_aug", action="store_true")
    parser.add_argument("--elr_beta", type=float, default=0.7)
    parser.add_argument("--elr_lambda", type=float, default=3.0)
    parser.add_argument("--elr_mixup_alpha", type=float, default=1.0)
    parser.add_argument("--elr_ema_alpha", type=float, default=0.997)
    parser.add_argument("--elr_ema_step", type=int, default=40000)
    parser.add_argument("--elr_ema_update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--elr_coef_step", type=int, default=0)
    parser.add_argument("--elr_warmup_epochs", type=int, default=0)
    parser.add_argument("--alim_prior", type=float, default=0.5)
    parser.add_argument("--alim_conf_ema_start", type=float, default=0.95)
    parser.add_argument("--alim_conf_ema_end", type=float, default=0.80)
    parser.add_argument("--dm_warmup_epochs", type=int, default=10)
    parser.add_argument("--dm_alpha", type=float, default=0.5)
    parser.add_argument("--dm_temp", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed_dataset is None:
        args.seed_dataset = args.seed
    if args.num_classes is None:
        args.num_classes = DATASET_SPECS[args.dataset].num_classes
    if args.experiment_name is None:
        args.experiment_name = (
            f"{args.method}_{args.dataset}_lpi{args.lpi}_slice{args.slice}"
            f"_{args.split_protocol}_seed{args.seed}_controlled"
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) / args.method / args.dataset / f"lpi{args.lpi}_slice{args.slice}_seed{args.seed}"
    write_run_header(args, out_dir)
    log_path = out_dir / "results.log"
    print(f"[RUN] out_dir={out_dir}", flush=True)
    print(f"[RUN] log_path={log_path}", flush=True)

    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        print(f"[ARGS] {json.dumps(vars(args), ensure_ascii=False, sort_keys=True)}", flush=True)
        print(f"[DEVICE] {device}", flush=True)
        train_set, train_loader, eval_train_loader, test_loader = make_loaders(args)
        if args.method == "dividemix":
            result = run_dividemix(args, device, train_set, train_loader, eval_train_loader, test_loader, out_dir)
        elif args.method == "elr_plus_full":
            result = run_elr_plus_full(args, device, train_set, train_loader, eval_train_loader, test_loader, out_dir)
        else:
            result = run_single_model(args, device, train_set, train_loader, eval_train_loader, test_loader, out_dir)
        print(f"[RESULT] {json.dumps(result, ensure_ascii=False, sort_keys=True)}", flush=True)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    print(f"[DONE] {args.experiment_name} -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
