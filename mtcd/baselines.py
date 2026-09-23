"""Image-differencing baselines, verbatim from the notebook (cell 12)."""
from __future__ import annotations

import numpy as np
import cv2
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from .data import LEVIR_MEAN, LEVIR_STD


def otsu(diff: np.ndarray) -> np.ndarray:
    """Otsu binarisation on a 2-D float map."""
    norm = cv2.normalize(diff, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    _, b = cv2.threshold(norm, 0, 1,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


@torch.no_grad()
def eval_diff_levir(loader, n=200):
    MEAN = np.array(LEVIR_MEAN, np.float32)
    STD  = np.array(LEVIR_STD,  np.float32)
    preds, labels, count = [], [], 0
    for i1, i2, m in tqdm(loader, desc='Image-Diff LEVIR'):
        for b in range(i1.size(0)):
            if count >= n: break
            a = i1[b].permute(1, 2, 0).numpy() * STD + MEAN   # denormalise
            c = i2[b].permute(1, 2, 0).numpy() * STD + MEAN
            diff = np.abs(c - a).mean(-1).astype(np.float32)
            preds.append(otsu(diff).ravel())
            labels.append(m[b].numpy().astype(int).ravel())
            count += 1
        if count >= n: break
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def eval_diff_onera(loader, n=100):
    preds, labels, count = [], [], 0
    for i1, i2, m in tqdm(loader, desc='Image-Diff ONERA'):
        for b in range(i1.size(0)):
            if count >= n: break
            a = i1[b].permute(1, 2, 0).numpy()   # (H,W,13) already normalised
            c = i2[b].permute(1, 2, 0).numpy()
            dv = c - a
            H, W, C = dv.shape
            score = np.abs(
                PCA(1).fit_transform(dv.reshape(-1, C))
            ).reshape(H, W).astype(np.float32)
            preds.append(otsu(score).ravel())
            labels.append(m[b].numpy().astype(int).ravel())
            count += 1
        if count >= n: break
    return np.concatenate(preds), np.concatenate(labels)