"""Evaluation helpers, verbatim from the notebook (cells 42-46)."""
from __future__ import annotations

import torch

from .config import CFG


@torch.no_grad()
def predict_one(mdl, i1, i2, dev, thr=0.5):
    mdl.eval()
    out = mdl(i1.unsqueeze(0).to(dev), i2.unsqueeze(0).to(dev))
    if isinstance(out, tuple): out = out[0]
    prob = torch.sigmoid(out.squeeze()).cpu().numpy()
    return (prob > thr).astype(int), prob


@torch.no_grad()
def fp_fn_rates(model, loader, dev, n=200):
    """Returns mean FP rate and FN rate over n samples."""

    tp_total = fp_total = fn_total = tn_total = 0
    count = 0
    for i1, i2, m in loader:
        out = model(i1.to(dev), i2.to(dev))
        if isinstance(out, tuple): out = out[0]
        pred  = (torch.sigmoid(out.squeeze(1)) > CFG['threshold']
                 ).cpu().numpy().astype(int)
        label = m.numpy().astype(int)
        for b in range(pred.shape[0]):
            if count >= n: break
            p = pred[b].ravel(); l = label[b].ravel()
            tp_total += ((p == 1) & (l == 1)).sum()
            fp_total += ((p == 1) & (l == 0)).sum()
            fn_total += ((p == 0) & (l == 1)).sum()
            tn_total += ((p == 0) & (l == 0)).sum()
            count += 1
        if count >= n: break
    total = tp_total + fp_total + fn_total + tn_total + 1e-9
    return fp_total / total, fn_total / total