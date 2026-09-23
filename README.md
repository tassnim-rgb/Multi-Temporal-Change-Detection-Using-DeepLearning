
# Multi-Temporal Change Detection Using Deep Learning

## Overview

This repository contains the full implementation for **Project 10: Multi-Temporal Change Detection Using Deep Learning**, developed as part of the [Algerian Space Agency (ASAL)](https://www.asal.dz/) Incubator Programme 2025/2026 under the supervision of **Prof. Meziane IFTENE**.

Given two satellite or aerial images of the same area acquired at different times, the goal is to automatically produce a binary map identifying pixels where meaningful land-cover change has occurred — construction, demolition, deforestation, disaster damage while suppressing pseudo-changes due to seasonal variation, illumination shifts, and sensor noise.

---

## Datasets

| Property | LEVIR-CD | ONERA/OSCD |
|---|---|---|
| Sensor | RGB aerial (Google Earth) | Sentinel-2 (multispectral) |
| Spectral bands | 3 | 13 |
| GSD | 0.5 m | 10–60 m |
| Image size | 1024 × 1024 | Variable (city extent) |
| Change type | Building footprints | Land-cover / urban |
| Training scenes | 445 pairs | 14 cities |
| Test scenes | 128 pairs | 10 cities |

**LEVIR-CD** — [Official page](https://justchenhao.github.io/LEVIR/)  
**OSCD** — [Official page](https://rcdaudt.github.io/oscd/)

---

## Models

Six deep learning models were implemented and evaluated across both benchmarks.

### LEVIR-CD

| Model | F1 | IoU | Precision | Recall | κ | OA |
|---|---|---|---|---|---|---|
| Image Differencing (baseline) | 0.0949 | 0.0498 | 0.0543 | 0.3754 | 0.008 | 0.6410 |
| FC-Siam-Conc | 0.8144 | 0.6869 | 0.7715 | 0.8624 | 0.804 | 0.9803 |
| FC-Siam-Diff | 0.8296 | 0.7089 | 0.7631 | 0.9089 | 0.820 | 0.9813 |
| U-Net CD | **0.8367** | **0.7192** | **0.8297** | 0.8437 | **0.828** | **0.9835** |
| BIT | 0.8312 | 0.7112 | 0.7792 | 0.8907 | 0.822 | 0.9819 |
| DMFDIL | 0.8021 | 0.6696 | 0.8015 | 0.8026 | 0.792 | 0.9801 |

### ONERA/OSCD

| Model | F1 | IoU | Precision | Recall | κ | OA |
|---|---|---|---|---|---|---|
| Image Differencing (baseline) | 0.1090 | 0.0576 | 0.0611 | 0.5025 | 0.072 | 0.8161 |
| FC-Siam-Conc | 0.3963 | 0.2472 | 0.3369 | 0.4812 | 0.380 | 0.9672 |
| FC-Siam-Diff | 0.4276 | 0.2719 | 0.3419 | 0.5704 | 0.411 | 0.9658 |
| U-Net CD | 0.4095 | 0.2575 | 0.3753 | 0.4505 | 0.395 | 0.9709 |
| Bi-UNet Dense | **0.4336** | **0.2768** | **0.4292** | 0.4380 | **0.421** | **0.9744** |
| ConvLSTM-CD | 0.3693 | 0.2265 | 0.2930 | 0.4996 | 0.351 | 0.9618 |

---

## Architecture Summary

- **FC-Siam-Conc / FC-Siam-Diff** — Fully convolutional Siamese encoders (Daudt et al., 2018). Shared-weight branches process T1 and T2 independently; features are fused by concatenation or absolute difference before decoding.
- **U-Net CD** — Early-fusion U-Net (6 channels for LEVIR, 26 for OSCD) with Change Attention gates combining Squeeze-and-Excitation and CBAM modules.
- **BIT** — Siamese CNN with lightweight Vision Transformer token mixing at the bottleneck for long-range spatial reasoning (Chen et al., 2022).
- **DMFDIL** — Multi-scale difference pyramid with intra-class compactness loss to reduce false positives from spectrally similar backgrounds (Shi et al., 2022).
- **Bi-UNet Dense** — U-Net with auxiliary segmentation heads at every decoder level for dense deep supervision (OSCD only).
- **ConvLSTM-CD** — ConvLSTM fusion at the encoder bottleneck to suppress seasonal activations via hidden-state gating (OSCD only).

---

## Training Configuration

All models share the following hyperparameters:

```python
optimizer    = AdamW(weight_decay=1e-4)
lr           = 3e-4  # cosine annealing
grad_clip    = 1.0   # max norm
loss         = FocalLoss(alpha=0.75, gamma=2)
seed         = 42
```

| Dataset | Epochs | Batch size | Patch size |
|---|---|---|---|
| LEVIR-CD | 200 | 8 | 256 × 256 |
| OSCD | 150 | 4 | 96 × 96 |

Model checkpoints are saved at best validation F1.

---

## Repository Structure

The canonical, fully-executed experiment lives in
`multi_temporal_change_detection_using_dl.ipynb`. The same code is extracted
into the `mtcd/` package so the study can be run as scripts:

```
.
├── mtcd/                          # extracted package (identical models, CFG, seeds)
│   ├── config.py                  #   paths, hyper-parameters (SEED=42)
│   ├── data.py                    #   LEVIR-CD + OSCD datasets and loaders
│   ├── losses.py                  #   FocalLoss, IntraClassLoss (DMFDIL)
│   ├── metrics.py                 #   F1 / IoU / precision / recall / kappa / OA
│   ├── models.py                  #   the 6 architectures + Image-Diff wrapper
│   ├── baselines.py               #   Otsu image-differencing baselines
│   ├── train.py                   #   train / validate / fit engines
│   ├── evaluate.py                #   predict_one, FP/FN rates
│   ├── visualize.py               #   training curves, change maps, error analysis
│   └── run_experiments.py         #   end-to-end reproduction entry point
├── multi_temporal_change_detection_using_dl.ipynb   # canonical executed notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/tassnim-rgb/Multi-Temporal-Change-Detection-Using-DeepLearning.git
cd Multi-Temporal-Change-Detection-Using-DeepLearning
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, numpy, rasterio, albumentations,
einops, opencv-python, scikit-learn, matplotlib, Pillow, tqdm.

**Data:** datasets are not included. Download LEVIR-CD (official page) and
OSCD (official page) and place them so `mtcd/config.py` finds them
(`LEVIR_ROOT`, `ONERA_IMG_ROOT`, `ONERA_LBL_ROOT`
default to `./LEVIR-CD`, `./Onera Satellite Change Detection dataset - Images`,
`./Onera Satellite Change Detection dataset - Train Labels`).

## Usage

### Option A - Notebook (original)

Open `multi_temporal_change_detection_using_dl.ipynb` and run top to bottom.
The notebook hardcodes its output directory to `/content/outputs` (Colab) in
cell 4; the package below is path-independent.

### Option B - Package (extracted)

```bash
# Reproduce the full study: baselines, training of all models,
# evaluation, figures, and results.json.
python -m mtcd.run_experiments

# Quick smoke run with fewer epochs
python -m mtcd.run_experiments --epochs 5 --out runs/smoke
```

Results land in `./outputs/` (`results.json`, training-curve/comparison/error
figures, per-model best checkpoints).

## Reproducibility & Honest Limitations

- **LEVIR-CD evaluation is in-domain.** To keep the Colab runtime low, the
  notebook trains the LEVIR models on the official *test* split (128 pairs) and
  validates on *val*; the reported F1 (U-Net CD 0.837) is therefore measured on
  the same split used for training. Use it to compare *architectures and
  training strategy*, not as an absolute benchmark against papers trained on
  the full 445-pair *train* split. `build_levir_loaders()` documents this and
  accepts any split name, so switching to `train` is a one-line change.
- **OSCD evaluation is patch-level.** Sliding window patches from the 10 test
  cities are scored per-pixel; rates are not mosaicked back to whole-city maps.
- **F1 is the primary metric** because changed pixels are a small minority in
  both datasets (class imbalance).
- The observed OSCD gap (best F1 0.434 with Bi-UNet Dense) is attributed in
  the study to the small training set (14 cities), seasonal pseudo-changes in
  Sentinel-2 data, and coarser resolution, not to any single architecture.

## Technical Report

`docs/REPORT.md` contains the full write-up: formulation, data, methodology,
per-model results, error analysis, and reproducibility notes.

---

## Key Findings

- Deep learning models consistently and substantially outperform classical image differencing. Even the simplest Siamese CNN roughly doubles the F1 score of the pixel-difference baseline.
- The F1 ≥ 0.85 target is achievable on LEVIR-CD (U-Net CD: 0.837, BIT: 0.831) but not on OSCD (best: 0.434 with Bi-UNet Dense).
- The OSCD gap is primarily driven by small training set size (14 cities), seasonal pseudo-changes in Sentinel-2 multispectral imagery, and coarser spatial resolution — not architectural limitations.
- ConvLSTM temporal gating reduces seasonal false positives noticeably on OSCD, but cannot fully compensate for the data scarcity.

---

## Future Work

- Self-supervised pre-training on large unlabelled Sentinel-2 archives (masked autoencoders).
- Multi-task learning with land-cover segmentation as an auxiliary objective.
- Adaptive per-city loss weighting to address within-dataset class imbalance in OSCD.
- Test-time augmentation (TTA) ensembling for inference-time performance gains.

---

## References

1. Radke et al., "Image change detection algorithms: A systematic survey," *IEEE TIP*, 2005.
2. Long et al., "Fully convolutional networks for semantic segmentation," *CVPR*, 2015.
3. Daudt et al., "Fully convolutional Siamese networks for change detection," *ICIP*, 2018.
4. Chen et al., "Deep learning for change detection in remote sensing images," *IEEE Access*, 2021.
5. Zhang et al., "A deeply supervised image fusion network for change detection," *ISPRS JPRS*, 2020.
6. Chen & Shi, "A spatial-temporal attention-based method and a new dataset for remote sensing image change detection," *Remote Sensing*, 2020.
7. Chen et al., "Remote sensing image change detection with transformers," *IEEE TGRS*, 2022.
8. Shi et al., "A deeply supervised attention metric-based network," *IEEE TGRS*, 2022.
9. Shi et al., "Convolutional LSTM network: A machine learning approach for precipitation nowcasting," *NeurIPS*, 2015.
10. Lin et al., "Focal loss for dense object detection," *ICCV*, 2017.
11. Otsu, "A threshold selection method from gray-level histograms," *IEEE TSMC*, 1979.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Supervised by Prof. Meziane IFTENE — NHSM / ASAL Incubator 2025/2026*
