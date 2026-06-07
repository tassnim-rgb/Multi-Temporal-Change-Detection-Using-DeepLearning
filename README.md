# Multi-Temporal-Change-Detection-Using-DeepLearning
Identifying meaningful ground events across time using siamense networks and spatial temporal analysis
Satellites now photograph every corner of the Earth on a near-daily basis, and
the archives they produce are growing faster than any human analyst could
possibly review. Change detection (CD) is the discipline that makes this flood
of imagery actionable: given two images of the same area taken at different
times, the goal is to automatically flag pixels where something meaningful has
changed on the ground. The applications are immediate and concrete:
• mapping deforestation in tropical and boreal biomes;
• tracking urban expansion in rapidly developing cities;
• assessingbuildingdamageafterearthquakesorfloodsbeforerescueteams
are deployed;
• monitoringcroprotationandseasonalland-usetransitionsinagricultural
regions.
The surge in freely available satellite data, particularly ESA’s Sentinel-2 con
stellation, has removed the cost barrier that once limited change detection to
government agencies. Anyone can now download metre-resolution imagery
for any city on Earth. The bottleneck has shifted from data acquisition to data
analysis, and that is exactly where deep learning enters the picture
# Multi-Temporal Change Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange)]()

---

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

```
.
├── data/
│   ├── levir/                  # LEVIR-CD dataset (not included)
│   └── oscd/                   # OSCD dataset (not included)
├── models/
│   ├── fc_siam_conc.py
│   ├── fc_siam_diff.py
│   ├── unet_cd.py
│   ├── bit.py
│   ├── dmfdil.py
│   ├── bi_unet_dense.py
│   └── convlstm_cd.py
├── datasets/
│   ├── levir_dataset.py
│   └── oscd_dataset.py
├── losses/
│   └── focal_loss.py
├── train.py
├── evaluate.py
├── infer.py
├── configs/
│   ├── levir_config.yaml
│   └── oscd_config.yaml
├── notebooks/
│   ├── levir_training.ipynb
│   └── oscd_training.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/change-detection-dl.git
cd change-detection-dl
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.x, torchvision, numpy, rasterio, albumentations, tqdm, scikit-learn, matplotlib.

---

## Usage

### Training

```bash
# LEVIR-CD
python train.py --config configs/levir_config.yaml --model unet_cd

# OSCD
python train.py --config configs/oscd_config.yaml --model convlstm_cd
```

### Evaluation

```bash
python evaluate.py --config configs/levir_config.yaml --model unet_cd --checkpoint checkpoints/best_unet_cd.pth
```

### Inference

```bash
python infer.py --t1 path/to/image_t1.tif --t2 path/to/image_t2.tif --model unet_cd --checkpoint checkpoints/best_unet_cd.pth --output change_map.png
```

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
