import argparse
import ast
import os
from pathlib import Path

import numpy as np


TARGET = Path(os.environ.get("SRSE_TRAIN_SCRIPT", Path(__file__).with_name("srse_persistent.py")))


def _load_functions(*names):
    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    needed = {"_normalize_np_rows", *names}
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in needed
    ]
    present = {node.name for node in selected}
    missing = set(names) - present
    if missing:
        raise AssertionError(f"missing helper function(s): {sorted(missing)}")
    mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"np": np, "argparse": argparse}
    exec(compile(mod, str(TARGET), "exec"), ns)
    return tuple(ns[name] for name in names)


def _load_function(name):
    return _load_functions(name)[0]


def test_candidate_membership_insert_mask_ignores_existing_candidates():
    mask_fn = _load_function("_candidate_membership_insert_mask")
    prior = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    scores = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.20, 0.60, 0.20],
            [0.30, 0.25, 0.45],
        ]
    )
    eligible = np.asarray([True, True, True])

    insert = mask_fn(prior, scores, eligible, margin=0.0)
    assert insert.tolist() == [False, False, True]

    stricter = mask_fn(prior, scores, eligible, margin=0.21)
    assert stricter.tolist() == [False, False, False]


def test_soft_mass_metrics_separate_recovery_and_damage():
    metrics_fn = _load_function("_soft_mass_state_metrics")
    labels = np.asarray([0, 1, 2])
    clean0 = np.asarray([True, True, False])
    noisy0 = np.asarray([False, False, True])
    base = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    work = np.asarray(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.4, 0.5],
            [0.1, 0.2, 0.7],
        ]
    )

    metrics = metrics_fn(base, work, labels, clean0, noisy0)
    assert abs(metrics["mass_rec_n"] - 0.7) < 1e-12
    assert abs(metrics["damage_mass"] - 0.4) < 1e-12


def test_fredis_moves_use_score_difference_not_raw_confidence():
    fredis_moves, = _load_functions("_fredis_candidate_moves")
    prior = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    scores = np.asarray(
        [
            [0.40, 0.39, 0.20, 0.01],  # label 1 is non-candidate and close to top -> refine in
            [0.70, 0.10, 0.15, 0.05],  # candidate label 1 is far below top -> disambiguate out
            [0.20, 0.35, 0.34, 0.11],  # non-candidate label 2 close to top -> refine in
        ]
    )
    refine, disamb = fredis_moves(
        prior,
        scores,
        eligible_mask=np.asarray([True, True, True]),
        refine_threshold=0.02,
        disamb_threshold=0.50,
    )

    assert refine.tolist() == [
        [False, True, False, False],
        [False, False, False, False],
        [False, False, True, False],
    ]
    assert disamb.tolist() == [
        [False, False, False, False],
        [False, True, False, False],
        [False, False, False, False],
    ]


def test_fredis_v2_uses_top_non_candidate_and_absolute_thresholds():
    fredis_moves, = _load_functions("_fredis_candidate_moves")
    prior = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ]
    )
    scores = np.asarray(
        [
            [0.10, 0.84, 0.05, 0.01],  # top non-candidate below add confidence -> no add
            [0.10, 0.86, 0.03, 0.01],  # add only top non-candidate label 1
            [0.90, 0.04, 0.03, 0.03],  # remove candidate label 1
        ]
    )

    refine, disamb = fredis_moves(
        prior,
        scores,
        eligible_mask=np.asarray([True, True, True]),
        refine_threshold=0.05,
        disamb_threshold=0.85,
        refine_min_conf=0.85,
        disamb_max_conf=0.05,
        top_non_candidate_only=True,
    )

    assert refine.tolist() == [
        [False, False, False, False],
        [False, True, False, False],
        [False, False, False, False],
    ]
    assert disamb.tolist() == [
        [False, False, False, False],
        [False, False, False, False],
        [False, True, False, False],
    ]


def test_irnet_correction_uses_top_non_candidate_when_tau_indicates_noisy():
    irnet_mask, = _load_functions("_irnet_candidate_correction")
    prior = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    scores = np.asarray(
        [
            [0.40, 0.45, 0.15],  # best non-candidate exceeds best candidate -> insert label 1
            [0.10, 0.70, 0.20],  # clean-like tau > 0 -> no insert
            [0.05, 0.30, 0.65],  # best non-candidate exceeds best candidate -> insert label 2
        ]
    )

    insert_mask, insert_labels, tau = irnet_mask(
        prior,
        scores,
        eligible_mask=np.asarray([True, True, True]),
        tau_boundary=0.0,
        min_non_candidate_conf=0.0,
    )

    assert insert_mask.tolist() == [True, False, True]
    assert insert_labels.tolist() == [1, 2, 2]
    assert tau[0] < 0 and tau[1] > 0 and tau[2] < 0


def test_irnet_v2_requires_high_non_candidate_confidence():
    irnet_mask, = _load_functions("_irnet_candidate_correction")
    prior = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scores = np.asarray(
        [
            [0.40, 0.45, 0.15],
            [0.10, 0.86, 0.04],
        ]
    )

    insert_mask, insert_labels, tau = irnet_mask(
        prior,
        scores,
        eligible_mask=np.asarray([True, True]),
        tau_boundary=0.0,
        min_non_candidate_conf=0.85,
    )

    assert insert_mask.tolist() == [False, True]
    assert insert_labels.tolist() == [1, 1]
    assert tau[0] < 0 and tau[1] < 0


def test_promotion_v2_uses_estimated_noise_and_non_candidate_label_space():
    promotion_candidates, = _load_functions("_persistent_promotion_candidates")
    native_prior = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    scores = np.asarray(
        [
            [0.96, 0.04, 0.00],  # all-label high confidence is candidate -> no non-candidate promotion
            [0.02, 0.96, 0.02],  # estimated-noise and high non-candidate -> promote label 1
            [0.02, 0.01, 0.97],  # high non-candidate but not estimated-noise -> no promote
        ]
    )
    promote_mask, pseudo, conf = promotion_candidates(
        native_prior,
        scores,
        active_mask=np.asarray([False, False, False]),
        estimated_noise_mask=np.asarray([True, True, False]),
        scope="srse_estimated_noise_highconf",
        label_space="non_candidate",
        threshold=0.95,
    )

    assert promote_mask.tolist() == [False, True, False]
    assert pseudo.tolist() == [1, 1, 2]
    assert abs(conf[1] - 0.96) < 1e-12


def test_high_linear_threshold_schedule_runs_from_095_to_085():
    threshold_fn, = _load_functions("_current_writeback_threshold")
    args = argparse.Namespace(
        source_update_start_epoch=21,
        model_warmup_epochs=20,
        epochs=501,
        source_update_schedule="linear",
        source_update_threshold=0.65,
        source_update_threshold_start=0.95,
        source_update_threshold_end=0.85,
    )

    assert threshold_fn(args, 19) is None
    assert abs(threshold_fn(args, 20) - 0.95) < 1e-12
    assert abs(threshold_fn(args, 500) - 0.85) < 1e-12


def test_pico_soft_target_update_keeps_candidate_constraint_when_requested():
    pico_update, = _load_functions("_pico_soft_target_update")
    prior = np.asarray([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    scores = np.asarray([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])

    constrained = pico_update(prior, scores, alpha=0.2, candidate_constrained=True)
    unconstrained = pico_update(prior, scores, alpha=0.2, candidate_constrained=False)

    assert constrained[0, 2] == 0.0
    assert constrained[0, 1] > constrained[0, 0]
    assert unconstrained[0, 2] > 0.0
