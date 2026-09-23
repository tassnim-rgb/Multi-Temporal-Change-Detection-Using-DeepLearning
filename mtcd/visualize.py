"""Visualization helpers, verbatim from the notebook (cells 38-46)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .data import LEVIR_MEAN, LEVIR_STD
from .metrics import compute_metrics
from .evaluate import predict_one
from .baselines import otsu
from sklearn.decomposition import PCA


LCOLORS = ['#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#009688']
OCOLORS = ['#2196F3', '#FF9800', '#E91E63', '#4CAF50', '#F44336']


def plot_training_curves(levir_hist, onera_hist, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle('Training Curves', fontsize=14, fontweight='bold')

    for col, (hist_dict, colors, ds) in enumerate([
            (levir_hist, LCOLORS, 'LEVIR-CD'),
            (onera_hist, OCOLORS, 'ONERA/OSCD')]):
        for row, key in enumerate(['loss', 'f1']):
            ax = axes[row, col]
            for (tag, h), c in zip(hist_dict.items(), colors):
                ax.plot(h[key], color=c, lw=1.8, label=tag.split('|')[-1])
            if key == 'f1':
                ax.axhline(0.85, color='red', ls='--', lw=1.2,
                           label='Target 0.85')
            ylabel = 'Focal Loss' if key == 'loss' else 'Val F1'
            ax.set_title(f'{ds} - {ylabel}', fontsize=11)
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            ax.legend(fontsize=8,
                      loc='lower right' if key == 'f1' else 'upper right')
            ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=130)
    plt.close(fig)


def plot_model_comparison(all_results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison - F1 & IoU on Test Set',
                 fontsize=13, fontweight='bold')

    for ax, prefix, title, colors in [
            (axes[0], 'LEVIR', 'LEVIR-CD',   LCOLORS),
            (axes[1], 'ONERA', 'ONERA/OSCD', OCOLORS)]:
        sub   = {k: v for k, v in all_results.items() if prefix in k}
        names = [k.split('|')[-1] if '|' in k else k for k in sub]
        f1s   = [v['f1']  for v in sub.values()]
        ious  = [v['iou'] for v in sub.values()]
        x = np.arange(len(names)); w = 0.35
        b1 = ax.bar(x-w/2, f1s,  w, label='F1',  color='#1565C0', alpha=.85)
        b2 = ax.bar(x+w/2, ious, w, label='IoU', color='#2E7D32', alpha=.85)
        ax.bar_label(b1, fmt='%.3f', fontsize=8, padding=2)
        ax.bar_label(b2, fmt='%.3f', fontsize=8, padding=2)
        ax.axhline(0.85, color='red', ls='--', lw=1.3, label='Target F1 = 0.85')
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20,
                                             ha='right', fontsize=9)
        ax.set_ylim(0, 1.08); ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'model_comparison.png', dpi=130)
    plt.close(fig)


def to_rgb(tensor, is_levir):
    MEAN = np.array(LEVIR_MEAN, np.float32)
    STD  = np.array(LEVIR_STD,  np.float32)
    if is_levir:
        img = tensor.permute(1, 2, 0).numpy() * STD + MEAN
    else:
        img = tensor[[2, 1, 0]].permute(1, 2, 0).numpy()   # B04/B03/B02
        img = (img - img.min()) / (img.max()-img.min()+1e-6)
    return np.clip(img, 0, 1)


def diff_baseline(i1, i2, is_levir):
    MEAN = np.array(LEVIR_MEAN, np.float32)
    STD  = np.array(LEVIR_STD,  np.float32)
    if is_levir:
        a = i1.permute(1, 2, 0).numpy()*STD+MEAN
        b = i2.permute(1, 2, 0).numpy()*STD+MEAN
        return otsu(np.abs(b-a).mean(-1).astype(np.float32))
    else:
        dv = (i2-i1).permute(1, 2, 0).numpy()
        H, W, C = dv.shape
        sc = np.abs(PCA(1).fit_transform(dv.reshape(-1, C))).reshape(H, W)
        return otsu(sc.astype(np.float32))


def show_change_maps(ds_name, mdl_dict, dataset, dev, idx=0, is_levir=True,
                     out_dir=None):
    i1, i2, mask = dataset[idx]
    label = mask.numpy().astype(int)
    models = list(mdl_dict.items())
    n = len(models)

    fig = plt.figure(figsize=(4*(n+3), 8))
    fig.suptitle(f'{ds_name} - Change Maps (sample #{idx})',
                 fontsize=12, fontweight='bold')
    gs = fig.add_gridspec(2, n+3, hspace=0.35, wspace=0.15)

    def _show(ax, img, title, **kw):
        ax.imshow(img, **kw); ax.set_title(title, fontsize=8); ax.axis('off')

    _show(fig.add_subplot(gs[0, 0]), to_rgb(i1, is_levir), 'T1')
    _show(fig.add_subplot(gs[0, 1]), to_rgb(i2, is_levir), 'T2')
    _show(fig.add_subplot(gs[0, 2]), label, 'Ground Truth',
          cmap='gray', vmin=0, vmax=1)
    _show(fig.add_subplot(gs[1, 0]), diff_baseline(i1, i2, is_levir),
          'Image Diff', cmap='hot', vmin=0, vmax=1)

    positions = [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                 (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    for (r, c), (tag, mdl) in zip(positions, models):
        pred, prob = predict_one(mdl, i1, i2, dev)
        short = tag.split('|')[-1]
        try:
            ax = fig.add_subplot(gs[r, c])
            _show(ax, prob, f'{short}\n(prob)', cmap='jet', vmin=0, vmax=1)
        except Exception:
            pass

    if out_dir is not None:
        plt.savefig(out_dir / f'{ds_name.replace("/", "_")}_maps.png',
                    dpi=130, bbox_inches='tight')
    plt.close(fig)


def error_map(pred, label):
    rgb = np.zeros((*pred.shape, 3), np.uint8)
    rgb[(pred == 1) & (label == 1)] = [255, 255, 255]   # TP white
    rgb[(pred == 1) & (label == 0)] = [220, 50, 50]     # FP red
    rgb[(pred == 0) & (label == 1)] = [ 50, 50, 220]    # FN blue
    # TN stays black
    return rgb


def plot_error_analysis(ds_name, mdl_dict, dataset, dev, n=4, is_levir=True,
                        out_dir=None):
    n = min(n, len(dataset))
    models = list(mdl_dict.items())
    nm = len(models)

    fig, axes = plt.subplots(n, nm+1, figsize=(4*(nm+1), 4*n))
    if n == 1: axes = axes[np.newaxis]
    fig.suptitle(
        f'{ds_name} - Error Analysis  '
        f'(White=TP  Red=FP  Blue=FN  Black=TN)',
        fontsize=11, fontweight='bold')

    for row in range(n):
        i1, i2, mask = dataset[row]
        label = mask.numpy().astype(int)

        # T1 with GT overlay
        vis = to_rgb(i1, is_levir)
        ov  = (vis*255).astype(np.uint8)
        ov[label == 1, 0] = 255
        ov[label == 1, 1] = (ov[label == 1, 1]*.4).astype(np.uint8)
        ov[label == 1, 2] = (ov[label == 1, 2]*.4).astype(np.uint8)
        axes[row, 0].imshow(ov)
        axes[row, 0].set_title('T1 + GT overlay', fontsize=8)
        axes[row, 0].axis('off')

        for col, (tag, mdl) in enumerate(models, 1):
            pred, _ = predict_one(mdl, i1, i2, dev)
            em = error_map(pred, label)
            m  = compute_metrics(pred.ravel(), label.ravel())
            axes[row, col].imshow(em)
            axes[row, col].set_title(
                f'{tag.split("|")[-1]}\n'
                f'F1={m["f1"]:.3f}  P={m["precision"]:.3f}  R={m["recall"]:.3f}',
                fontsize=7)
            axes[row, col].axis('off')

    # Legend
    patches = [
        mpatches.Patch(color='white', label='TP', linewidth=1,
                       edgecolor='gray'),
        mpatches.Patch(color=(220/255, 50/255, 50/255), label='FP'),
        mpatches.Patch(color=(50/255, 50/255, 220/255), label='FN'),
        mpatches.Patch(color='black', label='TN'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    if out_dir is not None:
        fn = out_dir / f'{ds_name.replace("/", "_")}_errors.png'
        plt.savefig(fn, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_fp_fn(fp_fn, all_models, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('False Positive Rate vs False Negative Rate by Model',
                 fontsize=13, fontweight='bold')

    for ax, prefix, title in [
            (axes[0], 'LEVIR', 'LEVIR-CD'),
            (axes[1], 'ONERA', 'ONERA/OSCD')]:
        sub = {k: v for k, v in fp_fn.items() if prefix in k}
        names = [k.split('|')[-1] if '|' in k else k for k in sub]
        fp_r   = [v[0]*100 for v in sub.values()]
        fn_r   = [v[1]*100 for v in sub.values()]
        x = np.arange(len(names)); w = 0.35
        b1 = ax.bar(x-w/2, fp_r, w, label='FP rate (%)',
                    color='#C62828', alpha=.85)
        b2 = ax.bar(x+w/2, fn_r, w, label='FN rate (%)',
                    color='#1565C0', alpha=.85)
        ax.bar_label(b1, fmt='%.2f%%', fontsize=7, padding=2)
        ax.bar_label(b2, fmt='%.2f%%', fontsize=7, padding=2)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20,
                                             ha='right', fontsize=9)
        ax.set_ylabel('Rate (% of all pixels)')
        ax.set_title(title, fontsize=11); ax.legend(); ax.grid(axis='y', alpha=.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'fp_fn_analysis.png', dpi=130)
    plt.close(fig)