"""Training engine, verbatim from the notebook (cells 30-32, 34)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG, OUT_DIR
from .losses import FocalLoss
from .metrics import compute_metrics


def train_epoch(model, loader, opt, crit, dev):
    """Standard training epoch for all models except DMFDIL and BiUNetDense."""

    model.train(); total = 0.0
    for i1, i2, m in loader:
        i1, i2, m = i1.to(dev), i2.to(dev), m.to(dev)
        opt.zero_grad()
        loss = crit(model(i1, i2), m)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += loss.item()
    return total / len(loader)


def train_epoch_dmfdil(model, loader, opt, crit, dev, icl_weight=0.1):
    """Training epoch for DMFDIL (main loss + intra-class loss)."""

    model.train(); total = 0.0
    for i1, i2, m in loader:
        i1, i2, m = i1.to(dev), i2.to(dev), m.to(dev)
        opt.zero_grad()
        out, icl = model(i1, i2, m)
        loss = crit(out, m) + icl_weight * icl
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += loss.item()
    return total / len(loader)


def train_epoch_biunet(model, loader, opt, crit, dev, aux_weight=0.3):
    """Training epoch for BiUNetDense (main + deep supervision losses)."""

    model.train(); total = 0.0
    for i1, i2, m in loader:
        i1, i2, m = i1.to(dev), i2.to(dev), m.to(dev)
        opt.zero_grad()
        main, auxs = model(i1, i2)
        loss = crit(main, m)
        for a in auxs:
            am = F.interpolate(a, size=m.shape[1:],
                               mode='bilinear', align_corners=False)
            loss = loss + aux_weight * crit(am.squeeze(1), m)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, dev, thr=0.5):
    model.eval(); preds, labels = [], []
    for i1, i2, m in loader:
        out = model(i1.to(dev), i2.to(dev))
        if isinstance(out, tuple): out = out[0]   # BiUNetDense eval guard
        prob = torch.sigmoid(out.squeeze(1))
        preds.append((prob > thr).cpu().numpy().ravel())
        labels.append(m.numpy().astype(int).ravel())
    return compute_metrics(np.concatenate(preds),
                           np.concatenate(labels))


def fit(model, tr_loader, va_loader, epochs, lr, wd, dev, tag,
        train_fn=None, out_dir=OUT_DIR):
    model = model.to(dev)
    crit  = FocalLoss(CFG['focal_gamma'], CFG['focal_alpha']).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=lr*0.01)
    hist  = {'loss': [], 'f1': [], 'iou': []}
    best_f1 = 0.0
    ckpt    = out_dir / f'{tag.replace("|","_")}_best.pth'
    _train  = train_fn or train_epoch
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs+1):
        l = _train(model, tr_loader, opt, crit, dev)
        v = validate(model, va_loader, dev)
        sched.step()
        hist['loss'].append(l); hist['f1'].append(v['f1'])
        hist['iou'].append(v['iou'])
        if v['f1'] > best_f1:
            best_f1 = v['f1']
            torch.save(model.state_dict(), ckpt)
        if ep % 10 == 0 or ep == 1:
            print(f'  [{tag}] ep {ep:3d}/{epochs}  '
                  f'loss={l:.4f}  F1={v["f1"]:.4f}  IoU={v["iou"]:.4f}')

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    hist['best_f1'] = best_f1
    print(f'  best val F1 = {best_f1:.4f}')
    return hist