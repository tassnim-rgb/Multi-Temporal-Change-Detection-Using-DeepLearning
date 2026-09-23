"""Losses, verbatim from the notebook (cells 6 and 24)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for change detection."""
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        # logits : (B,1,H,W) or (B,H,W) | targets : (B,H,W) float
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        bce   = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none')
        p_t   = torch.sigmoid(logits) * targets +                 (1 - torch.sigmoid(logits)) * (1 - targets)
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha * (1 - p_t) ** self.gamma * bce).mean()


class IntraClassLoss(nn.Module):
    """
    Intra-class compactness loss (DMFDIL).
    Pulls same-class feature vectors toward their class prototype.
    """
    def __init__(self):
        super().__init__()

    def forward(self, feat, mask):
        # feat : (B, C, H, W)  |  mask: (B, H, W) binary float
        B, C, H, W = feat.shape
        m0 = (1 - mask).unsqueeze(1)   # no-change  (B,1,H,W)
        m1 = mask.unsqueeze(1)         # change     (B,1,H,W)
        loss = torch.tensor(0.0, device=feat.device)
        for m in [m0, m1]:
            n  = m.sum() + 1e-6
            proto = (feat * m).sum(dim=[0, 2, 3], keepdim=True) / n
            loss += ((feat - proto) ** 2 * m).mean()
        return loss