"""Paths, seeds, and hyper-parameters.

Values mirror the notebook exactly (`CFG` in cell 4 of
`multi_temporal_change_detection_using_dl.ipynb`, SEED=42 in cell 2).

Note: the notebook hardcodes the output directory to `/content/outputs`
(Google Colab). This package writes to `PROJECT_ROOT/outputs` instead so it
runs anywhere.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset roots (change to your download locations) ────────────────────────
LEVIR_ROOT = PROJECT_ROOT / "LEVIR-CD"
ONERA_IMG_ROOT = PROJECT_ROOT / "Onera Satellite Change Detection dataset - Images"
ONERA_LBL_ROOT = PROJECT_ROOT / "Onera Satellite Change Detection dataset - Train Labels"

OUT_DIR = PROJECT_ROOT / "outputs"

# ── Hyper-parameters (identical to the notebook) ─────────────────────────────
CFG = dict(
    # LEVIR
    levir_patch  = 256,
    levir_batch  = 8,
    levir_epochs = 200,
    levir_lr     = 3e-4,
    # ONERA
    onera_patch  = 96,
    onera_batch  = 4,
    onera_epochs = 150,
    onera_lr     = 3e-4,
    # Shared
    weight_decay = 1e-4,
    focal_gamma  = 2.0,
    focal_alpha  = 0.75,
    base_ch      = 32,
    threshold    = 0.5,
)

SEED = 42


def seed_all(seed: int = SEED) -> None:
    """Mirror the notebook's seeding block."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")