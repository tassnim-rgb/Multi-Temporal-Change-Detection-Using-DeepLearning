"""End-to-end run: baselines, training of all models, evaluation, figures.

Mirrors the canonical notebook flow (cells 8 through 48) as a reproducible
script. Defaults reproduce the notebook exactly; flags only add convenience.

Usage:
    python -m mtcd.run_experiments                 # full run (defaults)
    python -m mtcd.run_experiments --epochs 20     # quick smoke run
    python -m mtcd.run_experiments --out runs/smoke
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import config
from .baselines import eval_diff_levir, eval_diff_onera
from .metrics import compute_metrics
from .models import (FCSiamConc, FCSiamDiff, UNetCD, BIT, DMFDIL,
                     BiUNetDense, ConvLSTMCD, ImgDiffModel,
                     parameter_summary)
from .train import fit, train_epoch_dmfdil, train_epoch_biunet, validate
from .evaluate import fp_fn_rates
from .visualize import (plot_training_curves, plot_model_comparison,
                        plot_fp_fn, show_change_maps, plot_error_analysis)


def as_plain(d: dict) -> dict:
    """Recursively convert numpy values to plain floats for JSON."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = as_plain(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) if isinstance(x, (np.floating, np.integer))
                      else x for x in v]
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs', type=int, default=None,
                    help='override both LEVIR and ONERA epochs '
                         '(default: CFG values 200 / 150)')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--out', type=str, default=None,
                    help='output directory (default: ./outputs)')
    args = ap.parse_args()

    config.seed_all(config.SEED)
    DEVICE = config.get_device() if args.device is None \
        else __import__('torch').device(args.device)
    out_dir = config.PROJECT_ROOT / (args.out or 'outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device : {DEVICE}')
    print(f'Output : {out_dir}')

    CFG = config.CFG
    if args.epochs:
        CFG = dict(CFG, levir_epochs=args.epochs, onera_epochs=args.epochs)

    # ── Loaders (notebook cell 8 / 10) ─────────────────────────────────────
    from .data import build_levir_loaders, build_onera_loaders
    lev_tr_loader, lev_va_loader, lev_te_loader = build_levir_loaders()
    on_tr_loader, on_va_loader = build_onera_loaders()

    # ── Baselines (cell 12) ────────────────────────────────────────────────
    lev_diff_p, lev_diff_l = eval_diff_levir(lev_te_loader)
    on_diff_p, on_diff_l = eval_diff_onera(on_va_loader)
    baseline_results = {
        'LEVIR|Image Diff': compute_metrics(lev_diff_p, lev_diff_l),
        'ONERA|Image Diff': compute_metrics(on_diff_p, on_diff_l),
    }
    for k, v in baseline_results.items():
        print(f'\n{k}')
        for metric, score in v.items():
            print(f'  {metric:12s}: {score:.4f}')

    print('\n--- Parameter summary ---')
    parameter_summary()

    # ── LEVIR models (cell 32) ─────────────────────────────────────────────
    B = CFG['base_ch']
    levir_models = {
        'LEVIR|FC-Siam-Conc': FCSiamConc(in_ch=3, base=B),
        'LEVIR|FC-Siam-Diff': FCSiamDiff(in_ch=3, base=B),
        'LEVIR|UNet-CD':      UNetCD    (in_ch=3, base=B),
        'LEVIR|BIT':          BIT       (in_ch=3, base=B),
        'LEVIR|DMFDIL':       DMFDIL    (in_ch=3, base=B),
    }

    levir_hist = {}
    for tag, mdl in levir_models.items():
        print(f'\n{"="*55}\n  {tag}\n{"="*55}')
        train_fn = train_epoch_dmfdil if 'DMFDIL' in tag else None
        levir_hist[tag] = fit(
            mdl, lev_tr_loader, lev_va_loader,
            epochs=CFG['levir_epochs'], lr=CFG['levir_lr'],
            wd=CFG['weight_decay'], dev=DEVICE,
            tag=tag, train_fn=train_fn, out_dir=out_dir)

    # ── ONERA models (cell 34) ─────────────────────────────────────────────
    onera_models = {
        'ONERA|FC-Siam-Conc':  FCSiamConc (in_ch=13, base=B),
        'ONERA|FC-Siam-Diff':  FCSiamDiff (in_ch=13, base=B),
        'ONERA|UNet-CD':       UNetCD     (in_ch=13, base=B),
        'ONERA|BiUNet-Dense':  BiUNetDense(in_ch=13, base=B),
        'ONERA|ConvLSTM-CD':   ConvLSTMCD (in_ch=13, base=B),
    }

    onera_hist = {}
    for tag, mdl in onera_models.items():
        print(f'\n{"="*55}\n  {tag}\n{"="*55}')
        train_fn = train_epoch_biunet if 'BiUNet' in tag else None
        onera_hist[tag] = fit(
            mdl, on_tr_loader, on_va_loader,
            epochs=CFG['onera_epochs'], lr=CFG['onera_lr'],
            wd=CFG['weight_decay'], dev=DEVICE,
            tag=tag, train_fn=train_fn, out_dir=out_dir)

    # ── Evaluate all (cell 36) ─────────────────────────────────────────────
    all_results = dict(baseline_results)
    for tag, mdl in levir_models.items():
        all_results[tag] = validate(mdl, lev_te_loader, DEVICE)
    for tag, mdl in onera_models.items():
        all_results[tag] = validate(mdl, on_va_loader, DEVICE)

    print(f'\n{"Model":<30} {"F1":>7} {"IoU":>7} {"Prec":>7} '
          f'{"Rec":>7} {"Kappa":>7} {"OA":>7}')
    print('-'*72)
    prev = ''
    for name, m in all_results.items():
        prefix = name.split('|')[0]
        if prefix != prev:
            print(); prev = prefix
        print(f'{name:<30} {m["f1"]:>6.4f} {m["iou"]:>7.4f} '
              f'{m["precision"]:>7.4f} {m["recall"]:>7.4f} '
              f'{m["kappa"]:>7.4f} {m["oa"]:>7.4f}')

    (out_dir / 'results.json').write_text(
        json.dumps(as_plain(all_results), indent=2))

    # ── Figures (cells 38-46) ──────────────────────────────────────────────
    plot_training_curves(levir_hist, onera_hist, out_dir)
    plot_model_comparison(all_results, out_dir)

    # the underlying datasets live on the loaders built above
    lev_dataset = lev_te_loader.dataset
    on_dataset = on_va_loader.dataset
    show_change_maps('LEVIR-CD', levir_models, lev_dataset, DEVICE,
                     idx=0, is_levir=True, out_dir=out_dir)
    show_change_maps('ONERA-OSCD', onera_models, on_dataset, DEVICE,
                     idx=0, is_levir=False, out_dir=out_dir)

    lev_all = {'LEVIR|Image Diff': ImgDiffModel(True)}
    lev_all.update(levir_models)
    on_all = {'ONERA|Image Diff': ImgDiffModel(False)}
    on_all.update(onera_models)

    plot_error_analysis('LEVIR-CD', lev_all, lev_dataset,
                        DEVICE, n=4, is_levir=True, out_dir=out_dir)
    plot_error_analysis('ONERA-OSCD', on_all, on_dataset,
                        DEVICE, n=4, is_levir=False, out_dir=out_dir)

    # ── FP/FN rates (cell 46) ──────────────────────────────────────────────
    fp_fn = {}
    for tag, mdl in {**lev_all, **on_all}.items():
        loader = lev_te_loader if 'LEVIR' in tag else on_va_loader
        fp, fn = fp_fn_rates(mdl, loader, DEVICE)
        fp_fn[tag] = (fp, fn)
    plot_fp_fn(fp_fn, {**lev_all, **on_all}, out_dir)

    # ── Final ranked results (cell 48) ─────────────────────────────────────
    print('\n  FINAL RANKED RESULTS')
    for prefix, ds in [('LEVIR', 'LEVIR-CD'), ('ONERA', 'ONERA/OSCD')]:
        print(f'\n  {ds}')
        sub = sorted(
            {k: v for k, v in all_results.items() if prefix in k}.items(),
            key=lambda x: x[1]['f1'], reverse=True)
        for i, (name, m) in enumerate(sub, 1):
            short = name.split('|')[-1] if '|' in name else name
            ok = ' TARGET' if m['f1'] >= 0.85 else ''
            print(f'  {i}. {short:<22} '
                  f'F1={m["f1"]:.4f}  IoU={m["iou"]:.4f}  '
                  f'Kappa={m["kappa"]:.4f}  {ok}')


if __name__ == '__main__':
    main()