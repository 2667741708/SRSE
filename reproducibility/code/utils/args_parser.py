# -*- coding: utf-8 -*-
"""SRSE reproducibility implementation."""
import argparse
import os
import sys
import random
import logging
import numpy as np
import torch


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




def setup_logger(log_dir, filename="run.log", is_master=False, to_console=False):
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
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (Only active if to_console is True)
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger



def parse_args():
    parser = argparse.ArgumentParser(description='SRSE training arguments with class-balanced reliable selection')
    
    parser.add_argument('--exp_name', type=str, default='SRSE_Run', help='Experiment name.')

    
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        choices=['CIFAR10', 'CIFAR100', 'CIFAR100H', 'Treeversity', 'Benthic', 'Plankton',])
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--out', type=str, default='./out_ultimate', help='Directory for output')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1], help='List of random seeds.')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')

    
    parser.add_argument('--pr', type=float, default=0.05, help='partial ratio (q)')
    parser.add_argument('--nr', type=float, default=0.5, help='noise ratio (eta)')
    parser.add_argument('--lpi', type=int, default=10, help='Labels Per Image (LPI) for crowdsource NPLL conversion')
    
    parser.add_argument('--network', type=str, default='R18', help='Network architecture (R18, R50)')
    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs.')
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
    
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Alpha for Mixup.')
    parser.add_argument('--lsr', type=float, default=0.5, help='[Compatibility only] Legacy label smoothing rate.')
    parser.add_argument('--consistency_weight', type=float, default=1.0, help='Weight for consistency loss.')

    
    parser.add_argument('--no_reliable_mixup', action='store_true', help='[Ablation] Disable Mixup on reliable set.')
    parser.add_argument('--no_rebalance', action='store_true', help='[Ablation] Disable class rebalancing on pseudo-labels.')
    parser.add_argument('--no_softmatch', action='store_true', help='[Ablation] Disable SoftMatch weighting (force weight=1.0).')
    parser.add_argument('--no_unreliable_mixup', action='store_true', help='[Ablation] Disable Mixup on unreliable set (use standard consistency).')

    
    parser.add_argument('--no_unreliable_training', action='store_true', help='[Ablation] COMPLETELY ignore unreliable set (Supervised Only).')

    
    parser.add_argument('--no_rectify', action='store_true', help='[Ablation] Disable Middleware Gating/Rectification.')

    
    parser.add_argument('--ema_alpha', type=float, default=0.999, help='EMA momentum factor (default: 0.999).')
    # ---------------------------------------------

    
    parser.add_argument('--k_val', type=int, default=15, help='k for knn')
    parser.add_argument('--delta', type=float, default=0.25, help='example selection quantile')

    parser.add_argument('--history_len', type=int, default=15, help='example selection quantile')
    
    parser.add_argument('--consensus_power', type=float, default=2.0, help='Power for consensus proportion in dynamic weight (default: 2.0)')
    parser.add_argument('--fix_dynamic_weight', action='store_true', help='[Ablation] Fix dynamic consistency weight to 1.0 (Disable curriculum)')
    # ----------------------------------------------

    parser.add_argument('--sim_mode_1', type=str, default='topology',  
                        choices=['topology', 'exp', 'daes', 'none','topology_daes'],   
                        help='Similarity measure for Stage 1')

    parser.add_argument('--sim_mode_2', type=str, default='daes',      
                        choices=['topology', 'exp', 'daes', 'none', 'topology_daes'],   
                        help='Similarity measure for Stage 2')

    
    parser.add_argument('--topology_rel_mode', type=str, default='masked_entropy',
                        choices=['masked_entropy', 'kl', 'agree'],
                        help='[Topology] Reliability score mode. Use kl/agree to support one-hot labels.')
    parser.add_argument('--topology_rel_gamma', type=float, default=2.0,
                        help='[Topology] Penalty strength for low-consensus nodes (default: 2.0).')
    parser.add_argument('--topology_rel_eps', type=float, default=1e-12,
                        help='[Topology] Epsilon for numerical stability.')

    parser.add_argument('--warmup_epochs', type=int, default=250, help='Epochs for linear LR warmup.')

    
    parser.add_argument('--detailed_log', action='store_true', help='Enable detailed diagnostic logging.')
    
    parser.add_argument('--gating_start_ratio', type=float, default=0.2,
                        help='[Gating] Ratio of epochs before teacher gating starts (default: 0.2, means start at 20%% epoch).')
    parser.add_argument('--gating_max_alpha', type=float, default=1.0,
                        help='[Gating] Max influence of teacher (0.0 to 1.0). Set <1.0 to always keep some geometry signal.')

    
    parser.add_argument('--daes_clamp', type=float, default=0.25,
                        help='[DAES] Max temperature clamp (Anti-oversmoothing lock). Lower is sharper.')
    parser.add_argument('--daes_entropy_weight', type=float, default=0.2,
                        help='[DAES] Sensitivity to local entropy (tau = base + weight * H).')
    parser.add_argument('--daes_sharpening_power', type=float, default=2.0,
                        help='[DAES] Sharpening power for local mean calculation (default=1.0).')

    
    parser.add_argument('--daes_spatial_temp', type=float, default=0.5,
                        help='[DAES] Temperature for spatial weighting (default: 0.5).')
    parser.add_argument('--daes_base_tau', type=float, default=0.1,
                        help='[DAES] Base temperature for affinity matrix (default: 0.1).')
    parser.add_argument('--daes_entropy_coeff', type=float, default=0.5,
                        help='[DAES] Coefficient for entropy-based temperature adjustment (default: 0.5).')
    parser.add_argument('--daes_sim_power', type=float, default=2.0,
                        help='[DAES] Power to raise similarity to (default: 2.0).')

    
    parser.add_argument('--daes_topology_ref_mode', type=str, default='hard', choices=['gated', 'hard'],
                        help='[DAES] Topology reference mode: "gated" (use teacher confidence gated signal) or "hard" (use raw hard signal).')

    
    parser.add_argument('--knn_heads', type=int, default=4,
                        help='[KNN] Number of  heads for metric learning (Robustness).')

    

    
    


    
    parser.add_argument('--enable_knn1_model_fuse', action='store_true',
                        help='[EXP9] Enable model-geometry fusion with KNN1 scores before candidate projection.')

    parser.add_argument('--fusion_mode', type=str, default='weighted_sum',
                        choices=['geometric', 'weighted_sum'],
                        help='Fusion mode for model prediction and KNN scores (default: weighted_sum)')

    # ===========================================================================
    
    # ===========================================================================
    parser.add_argument('--kl_self_mode', type=str, default='with_self',
                        choices=['with_self', 'no_self'],
                        help='[Ablation/KL] Self-node inclusion in KL reliability score.')

    # ===========================================================================
    
    
    
    
    
    
    # ===========================================================================
    parser.add_argument('--adap_rel_gamma', type=float, default=2.0,
                        help='[AdapFuse] Decay factor gamma for reliability score r_i=exp(-gamma*(CE/logC)^2). '
                             'Higher gamma makes the gate more aggressive.')
    parser.add_argument('--adap_rel_eps', type=float, default=1e-12,
                        help='[AdapFuse] Numerical stability epsilon for CE computation.')
    parser.add_argument('--adap_ce_direction', type=str, default='knn_over_model',
                        choices=['knn_over_model', 'model_over_knn'],
                        help='[AdapFuse] CE direction for reliability scoring.\n'
                             '  knn_over_model (default): ce=-sum p_knn2*log(p_model). '
                             'Low r_i when model is random (early epochs) -> prior dominates -> '
                             'candidate-set true-label info guides KNN (natural curriculum).\n'
                             '  model_over_knn: ce=-sum p_model*log(p_knn2). '
                             'Early: measures KNN entropy, rewards already-sharp KNN distributions.')

    # ===========================================================================
    
    
    
    
    
    # ===========================================================================
    parser.add_argument('--model_warmup_epochs', type=int, default=10,
                        help='[ProgFuse] Number of warmup epochs for model prediction weight. '
                             'w_model = max_w_model * min(1.0, epoch / model_warmup_epochs). '
                             'At epoch 0, w_model=0 -> p_model_effective = p_knn1 (pure first-pass KNN). '
                             'At epoch >= model_warmup_epochs, w_model=max_w_model -> capped model/KNN mixture.')
    
    parser.add_argument('--max_w_model', type=float, default=0.5,
                        help='Maximum value for w_model (default: 0.5).')

    # ===========================================================================
    
    
    parser.add_argument("--adaptive_prop_depth", action="store_true",
                        help="[AdapDepth] Automatically truncate propagation (skip Stage 3) if over-sharpening is detected (high consensus but low reliable volume).")
    parser.add_argument("--expected_rel_ratio", type=float, default=0.5,
                        help="[AdapDepth] Expected minimum ratio of reliable samples (default: 0.5). Used with --adaptive_prop_depth.")
    
    # ===========================================================================

    return parser.parse_args()
