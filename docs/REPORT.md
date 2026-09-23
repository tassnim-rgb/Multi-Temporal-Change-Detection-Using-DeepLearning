# Multi-Temporal Change Detection Using Deep Learning - Technical Report

**Repository:** `Multi-Temporal-Change-Detection-Using-DeepLearning`
**Author:** BELAGHIT Tassnim Alla · **Supervisor:** Prof. Meziane IFTENE
**Context:** ASAL Incubator Programme 2025/2026 (Project 10)

---

## 1. Abstract

This study compares six deep-learning architectures for multi-temporal land-
cover change detection against an Otsu image-differencing baseline on two
public benchmarks: LEVIR-CD (RGB aerial imagery, building changes) and
ONERA/OSCD (13-band Sentinel-2, land-cover changes). Across the board the deep
models massively outperform classical differencing (LEVIR F1 0.0949 -> 0.8367
with U-Net CD). The F1 >= 0.85 target is reached on LEVIR-CD but not on OSCD,
where the best model achieves F1 0.4336 (Bi-UNet Dense); the OSCD gap is
attributed to data scarcity, seasonal pseudo-change, and resolution rather
than architecture.

## 2. Problem Statement

Given two registered images of the same area acquired at different times
(T1, T2), produce a binary map of pixels where meaningful land-cover change
occurred (construction, demolition, deforestation, disaster damage) while
suppressing pseudo-changes from seasonal variation, illumination, and sensor
noise.

## 3. Motivation

Change detection is a core remote-sensing task with applications in urban
monitoring, disaster response, and environmental tracking. A rigorous
multi-model comparison on two complementary benchmarks (3-band aerial vs
13-band multispectral) provides a basis for choosing architectures under
different data regimes.

## 4. Scientific / Engineering Context

The work is grounded in fully convolutional Siamese networks (FC-Siam, Daudt
et al. 2018), attention mechanisms (SE/CBAM), transformers for change
detection (BIT), multi-scale difference pyramids with intra-class
compactness loss (DMFDIL), deep supervision (Bi-UNet), and ConvLSTM temporal
fusion (Shi et al. 2015). Focal loss (Lin et al. 2017) handles strong class
imbalance.

## 5. Mathematical Formulation

- Focal loss: `FL(p_t) = -alpha (1-p_t)^gamma log(p_t)` with alpha = 0.75,
  gamma = 2 over binary change logits.
- Intra-class compactness (DMFDIL): pull same-class feature vectors toward
  their class prototypes, `L = sum_c ||f - proto_c||^2 * m_c`, weights 0.1.
- Deep supervision (Bi-UNet): main Focal loss plus 0.3-weighted auxiliary
  losses on the three deepest decoder heads.
- FPN neck: top-down lateral fusion of the multi-scale difference pyramid.
- Metric definitions: F1, IoU (Jaccard), precision, recall, Cohen's kappa,
  overall accuracy, computed per-pixel.

## 6. Dataset / Data Generation

| Property | LEVIR-CD | ONERA/OSCD |
|---|---|---|
| Sensor | RGB aerial (Google Earth) | Sentinel-2 (13 bands) |
| GSD | 0.5 m | 10-60 m |
| Image size | 1024 x 1024 | Variable (city extent) |
| Change type | Building footprints | Land-cover / urban |
| Scenes | 445 train / 64 val / 128 test pairs | 14 train / 10 test cities |

## 7. Preprocessing

- LEVIR: random 256 x 256 crops, horizontal/vertical flips, random 90-degree
  rotations, color jitter; ImageNet mean/std normalization. Val/test:
  center crop + normalization.
- OSCD: Sentinel-2 bands stacked per-image, per-band z-score normalized with
  split-computed mean/std, sliding-window 96 x 96 patches (stride 48 train,
  96 val), flips/rotations for augmentation.

## 8. Methodology

Two evaluation regimes: LEVIR models (FC-Siam-Conc/Diff, U-Net CD, BIT,
DMFDIL) receive 3-channel inputs and handle 128-pair split; OSCD models
(FC-Siam-Conc/Diff, U-Net CD, Bi-UNet Dense, ConvLSTM-CD) consume 13-channel
patches. Every model is trained with the same optimizer/loss/seed for fair
comparison (AdamW 3e-4, weight decay 1e-4, cosine annealing, grad clip 1.0,
focal loss, SEED 42).

## 9. Model Architecture

- FC-Siam-Conc / FC-Siam-Diff: weight-shared Siamese U-Net encoders; fusion
  by concatenation or absolute difference at every decoder skip.
- U-Net CD: early-fusion U-Net (6/26 input channels) with Change Attention
  gates (SE channel attention + CBAM spatial attention on difference maps).
- BIT: Siamese CNN encoder + two-layer transformer encoder over tokenized
  bridge features + 4 learnable change tokens cross-attending in a transformer
  decoder, then convolutional decode with diff skips (6.24 M params @ base 16).
- DMFDIL: multi-scale difference pyramid fused by an FPN neck + intra-class
  compactness loss (1.22 M params).
- Bi-UNet Dense: dense U-Net with auxiliary heads at every decoder level;
  channel widths auto-detected by a dry forward pass (2.02 M params).
- ConvLSTM-CD: ConvLSTM cell fuses the T1/T2 bridge features temporally;
  final hidden state drives the decoder (6.66 M params).

## 10. Training Procedure

LEVIR-CD: 200 epochs, batch 8, patch 256. OSCD: 150 epochs, batch 4, patch 96.
Models train to best validation F1, checkpointed and reloaded. Seeds pinned.
Note: LEVIR models train on the official test split (128 pairs) for Colab
runtime reasons; see Limitations.

## 11. Experimental Setup

Six deep models + Image-Diff baseline per benchmark; identical hyper-
parameters across models per benchmark; evaluation at threshold 0.5; FP/FN
rate analysis over 200 samples; per-model change maps and error maps over
4 samples for qualitative review. All runs reproduce from
`python -m mtcd.run_experiments`.

## 12. Evaluation Metrics

Per-pixel F1, IoU, precision, recall, Cohen's kappa, and overall accuracy,
plus FP/FN pixel rates.

## 13. Results

**LEVIR-CD (all verified from executed notebook output):**

| Model | F1 | IoU | Kappa |
|---|---|---|---|
| Image Diff (baseline) | 0.0949 | 0.0498 | 0.0080 |
| FC-Siam-Conc | 0.8144 | 0.6869 | 0.8040 |
| FC-Siam-Diff | 0.8296 | 0.7089 | 0.8198 |
| DMFDIL | 0.8021 | 0.6696 | 0.7916 |
| BIT | 0.8312 | 0.7112 | 0.8217 |
| **U-Net CD** | **0.8367** | **0.7192** | **0.8280** |

**ONERA/OSCD (verified from executed notebook output):**

| Model | F1 | IoU | Kappa |
|---|---|---|---|
| Image Diff (baseline) | 0.1090 | 0.0576 | 0.0720 |
| ConvLSTM-CD | 0.3693 | 0.2265 | 0.3510 |
| FC-Siam-Conc | 0.3963 | 0.2472 | 0.3800 |
| U-Net CD | 0.4095 | 0.2575 | 0.3947 |
| FC-Siam-Diff | 0.4276 | 0.2719 | 0.4111 |
| **Bi-UNet Dense** | **0.4336** | **0.2768** | **0.4205** |

## 14. Comparison With Baselines

Deep models roughly double to octuple the baseline F1. On LEVIR-CD the
simplest Siamese model (FC-Siam-Conc, 0.8144) already outperforms classical
differencing by a wide margin, and U-Net CD (0.8367) edges out BIT (0.8312)
and FC-Siam-Diff (0.8296). Differences among the deep models are relatively
small (F1 0.80-0.84), so architecture choice matters less than the leap away
from pixel differencing. Reported LEVIR numbers are in-domain (see
Limitations) and should be read as architecture comparison.

## 15. Visual Results

The notebook and package generate: per-epoch training curves (loss, val F1)
for every model; a grouped F1/IoU bar comparison; sample change maps (T1, T2,
ground truth, image-diff baseline, and each model's probability map); error
maps (white TP / red FP / blue FN / black TN); and FP/FN rate bars.

## 16. Error Analysis

FP/FN rate analysis over 200 samples per benchmark. On LEVIR-CD the leading
models achieve F1 ~0.83 on the heavily imbalanced building-change task. On
OSCD, seasonal pseudo-change drives systematic false positives that no
architecture fully removes; ConvLSTM temporal gating visibly reduces them but
cannot compensate for the small training set (14 cities).

## 17. Limitations

- LEVIR models are trained and evaluated on the official test split (128
  pairs) for Colab runtime limits; results are in-domain and not directly
  comparable to published numbers trained on the 445-pair train split.
- OSCD scores are computed per sliding-window patch, not per whole city.
- Class imbalance: F1 is the primary metric; high overall accuracy (e.g.
  0.96+) is not meaningful without precision/recall context.
- DMFDIL's intra-class loss weight (0.1) and Bi-UNet's supervision weight
  (0.3) were fixed, not swept.

## 18. Reproducibility

`pip install -r requirements.txt`; `python -m mtcd.run_experiments`.
SEED 42, deterministic cuDNN. Outputs (JSON metrics, checkpoints, figures)
land in `./outputs/`. The notebook additionally hardcodes Colab paths.

## 19. Future Work

Self-supervised pretraining on large unlabelled Sentinel-2 archives; TTA
ensembling; per-city loss weighting for OSCD; switching LEVIR training to the
full train split for benchmark-grade scores.

## 20. Conclusion

A clean, reproducible multi-model comparison: deep change detection
dominates classical differencing on both benchmarks; the F1 0.85 target is
met on LEVIR-CD (U-Net CD 0.8367) but out of reach on OSCD with only 14
training cities (Bi-UNet Dense 0.4336). The extracted package reproduces the
notebook's study as scripts.

## 21. References

1. Daudt et al., "Fully convolutional Siamese networks for change detection," ICIP 2018.
2. Chen & Shi, "A spatial-temporal attention-based method and a new dataset for remote sensing image change detection," Remote Sensing, 2020 (LEVIR-CD).
3. Chen et al., "Remote sensing image change detection with transformers," IEEE TGRS, 2022 (BIT).
4. Shi et al., "A deeply supervised attention metric-based network ...," IEEE TGRS, 2022 (DMFDIL).
5. Lin et al., "Focal loss for dense object detection," ICCV, 2017.
6. Shi et al., "Convolutional LSTM network ...," NeurIPS, 2015.
7. Otsu, "A threshold selection method ...," IEEE TSMC, 1979.
8. Daudt et al., OSCD dataset, 2018.