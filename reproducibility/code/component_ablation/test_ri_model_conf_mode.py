#!/usr/bin/env python3
"""Smoke-test the r_i model-confidence mode used by the SRSE ablation script."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


def load_module():
    script_env = os.environ.get("SRSE_SCRIPT_PATH")
    if script_env is None:
        script_env = str(Path(__file__).with_name("srse_ablation.py"))
    script_path = Path(script_env).resolve()
    spec = importlib.util.spec_from_file_location("srse_train_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_model_confidence_for_ri() -> None:
    module = load_module()
    fn = module.compute_model_confidence_for_ri

    p_model = torch.tensor(
        [
            [0.05, 0.80, 0.15],
            [0.10, 0.20, 0.70],
        ],
        dtype=torch.float32,
    )
    omega = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    max_conf = fn(p_model, omega, mode="max")
    prior_mass_conf = fn(p_model, omega, mode="prior_mass")

    assert torch.allclose(max_conf, torch.tensor([0.80, 0.70])), max_conf
    assert torch.allclose(prior_mass_conf, torch.tensor([0.20, 0.20])), prior_mass_conf
    assert prior_mass_conf[0] < max_conf[0]


if __name__ == "__main__":
    test_compute_model_confidence_for_ri()
