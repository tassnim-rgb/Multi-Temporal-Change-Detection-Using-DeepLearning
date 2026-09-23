"""Evaluation metrics, verbatim from the notebook (cell 6)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (f1_score, jaccard_score,
                             precision_score, recall_score,
                             cohen_kappa_score)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict:
    p = preds.ravel().astype(int)
    l = labels.ravel().astype(int)
    return dict(
        f1        = f1_score(l, p, zero_division=0),
        precision = precision_score(l, p, zero_division=0),
        recall    = recall_score(l, p, zero_division=0),
        iou       = jaccard_score(l, p, zero_division=0),
        kappa     = cohen_kappa_score(l, p) if len(np.unique(l)) > 1 else 0.0,
        oa        = float(np.mean(p == l)),
    )