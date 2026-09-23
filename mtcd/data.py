"""Datasets and preprocessing, verbatim from the notebook (cells 8 and 10)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import CFG, LEVIR_ROOT, ONERA_IMG_ROOT, ONERA_LBL_ROOT

# ── LEVIR-CD ─────────────────────────────────────────────────────────────────
LEVIR_MEAN = (0.485, 0.456, 0.406)
LEVIR_STD = (0.229, 0.224, 0.225)


def levir_train_aug(patch):
    return A.Compose([
        A.RandomCrop(patch, patch),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.1, hue=0.05, p=0.4),
        A.Normalize(mean=LEVIR_MEAN, std=LEVIR_STD),
        ToTensorV2(),
    ], additional_targets={'image2': 'image'})


def levir_val_aug(patch):
    return A.Compose([
        A.CenterCrop(patch, patch),
        A.Normalize(mean=LEVIR_MEAN, std=LEVIR_STD),
        ToTensorV2(),
    ], additional_targets={'image2': 'image'})


class LEVIRDataset(Dataset):
    """
    Returns
    img_t1 : (3, H, W) float32  : T1 image, ImageNet-normalised
    img_t2 : (3, H, W) float32  : T2 image
    mask   : (H, W)   float32  : binary label {0, 1}
    """
    def __init__(self, root: Path, split: str, patch: int, aug):
        self.dir = root / split
        self.aug = aug
        self.ids = sorted(p.stem
                          for p in (self.dir / 'A').glob('*.png'))
        assert self.ids, f'No images found under {self.dir}/A'
        print(f'  LEVIR [{split:5s}] : {len(self.ids)} pairs')

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        n  = self.ids[idx]
        i1 = np.array(Image.open(self.dir/'A'    /f'{n}.png').convert('RGB'))
        i2 = np.array(Image.open(self.dir/'B'    /f'{n}.png').convert('RGB'))
        m  = np.array(Image.open(self.dir/'label'/f'{n}.png').convert('L'))
        m  = (m > 127).astype(np.uint8)
        out = self.aug(image=i1, image2=i2, mask=m)
        return out['image'], out['image2'], out['mask'].float()


def build_levir_loaders(root: Path = LEVIR_ROOT, patch: int = None,
                        batch: int = None):
    """Mirror the notebook loader construction.

    NOTE: the notebook trains on the LEVIR-CD ``test`` split (128 pairs) to
    keep Colab runtime low and evaluates on the same split. This is preserved
    intentionally; see README "Limitations".
    """
    patch = patch or CFG['levir_patch']
    batch = batch or CFG['levir_batch']

    lev_tr = LEVIRDataset(root, 'test', patch, levir_train_aug(patch))
    lev_va = LEVIRDataset(root, 'val',  patch, levir_val_aug(patch))
    lev_te = LEVIRDataset(root, 'test', patch, levir_val_aug(patch))

    lev_tr_loader = torch.utils.data.DataLoader(
        lev_tr, batch, shuffle=True,  num_workers=0, pin_memory=True)
    lev_va_loader = torch.utils.data.DataLoader(
        lev_va, batch, shuffle=False, num_workers=0, pin_memory=True)
    lev_te_loader = torch.utils.data.DataLoader(
        lev_te, 1, shuffle=False, num_workers=0)
    return lev_tr_loader, lev_va_loader, lev_te_loader


# ── ONERA / OSCD ─────────────────────────────────────────────────────────────
# Official OSCD train / test city split (Daudt et al. 2018)
TRAIN_CITIES = [
    'abudhabi', 'beirut',      'bercy',    'bordeaux',   'cupertino',
    'hongkong', 'lasvegas',    'milano',   'montpellier','mumbai',
    'norcia',   'paris',       'rennes',   'saclay_e',
]
TEST_CITIES = [
    'aguasclaras', 'beihai', 'brasilia', 'chongqing', 'dubai',
    'seoul',       'tokyo',  'tyrol-e',  'tyrol-w',
]

# Per-band Sentinel-2 mean / std (computed on OSCD training set)
# Band order: B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12 B01
S2_MEAN = np.array([1153.7, 1100.9, 1040.7, 1143.1, 1371.8, 1568.9,
                    1553.0, 1741.1, 1363.5,  895.6, 1175.6, 1246.4,
                    1280.2], dtype=np.float32)
S2_STD = np.array([ 321.1,  378.3,  508.0,  400.4,  395.3,  403.9,
                    434.8,  434.9,  432.2,  375.1,  344.0,  383.9,
                    357.5], dtype=np.float32)


def load_s2(city_dir: Path) -> np.ndarray:
    """
    Load & stack 13 Sentinel-2 bands from imgs_1_rect / imgs_2_rect.
    Each .tif in the folder is one band.  Bands are sorted by filename
    (B01 ... B12, B8A) so the order is consistent across cities.
    Returns (H, W, 13) float32, per-band normalised.
    """
    tifs = sorted(city_dir.glob('*.tif'))[:13]
    if not tifs:
        raise FileNotFoundError(f'No .tif files found in {city_dir}')
    bands = []
    for t in tifs:
        with rasterio.open(t) as src:
            bands.append(src.read(1).astype(np.float32))
    while len(bands) < 13:          # pad to 13 if fewer bands present
        bands.append(np.zeros_like(bands[0]))
    H, W = bands[0].shape
    # Upsample all bands to the resolution of band 0 (10 m)
    stack = np.stack([
        cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
        for b in bands
    ], axis=-1)                      # (H, W, 13)
    return ((stack - S2_MEAN) / (S2_STD + 1e-6)).astype(np.float32)


class ONERADataset(Dataset):
    """
    Sliding-window patch dataset for ONERA/OSCD.
    Returns
    img_t1 : (13, H, W) float32  : T1 patch, per-band normalised
    img_t2 : (13, H, W) float32  : T2 patch
    mask   : (H, W)    float32  : binary change label {0, 1}
    """
    def __init__(self, cities: List[str],
                 patch=96, stride=48, augment=False,
                 img_root: Path = ONERA_IMG_ROOT,
                 lbl_root: Path = ONERA_LBL_ROOT):
        self.patch   = patch
        self.augment = augment
        self.samples = []

        for city in cities:
            t1d = img_root / city / 'imgs_1_rect'
            t2d = img_root / city / 'imgs_2_rect'
            cmp = lbl_root / city / 'cm' / 'cm.png'

            if not t1d.exists():
                print(f'  [skip] {city} - {t1d} not found')
                continue
            if not cmp.exists():
                print(f'  [skip] {city} - label {cmp} not found')
                continue

            img1 = load_s2(t1d)
            img2 = load_s2(t2d)
            cm   = (np.array(Image.open(cmp).convert('L')) > 127
                    ).astype(np.uint8)
            H, W = cm.shape
            for y in range(0, H - patch + 1, stride):
                for x in range(0, W - patch + 1, stride):
                    self.samples.append((
                        img1[y:y+patch, x:x+patch],
                        img2[y:y+patch, x:x+patch],
                        cm  [y:y+patch, x:x+patch],
                    ))
        print(f'  ONERA [{len(cities)} cities] : {len(self.samples)} patches')

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        i1, i2, m = self.samples[idx]
        i1, i2, m = i1.copy(), i2.copy(), m.copy()
        if self.augment:
            if random.random() > .5:
                i1=i1[::-1]; i2=i2[::-1]; m=m[::-1]
            if random.random() > .5:
                i1=i1[:,::-1]; i2=i2[:,::-1]; m=m[:,::-1]
            k = random.randint(0, 3)
            i1=np.rot90(i1,k); i2=np.rot90(i2,k); m=np.rot90(m,k)
        t1 = torch.from_numpy(i1.copy().transpose(2,0,1))
        t2 = torch.from_numpy(i2.copy().transpose(2,0,1))
        mk = torch.from_numpy(m.copy().astype(np.float32))
        return t1, t2, mk


def build_onera_loaders(patch: int = None, batch: int = None):
    """Mirror the notebook loader construction."""
    patch = patch or CFG['onera_patch']
    batch = batch or CFG['onera_batch']

    on_tr = ONERADataset(TRAIN_CITIES, patch, stride=48, augment=True)
    on_va = ONERADataset(TEST_CITIES,  patch, stride=96, augment=False)

    on_tr_loader = torch.utils.data.DataLoader(
        on_tr, batch, shuffle=True,  num_workers=0, pin_memory=True)
    on_va_loader = torch.utils.data.DataLoader(
        on_va, batch, shuffle=False, num_workers=0, pin_memory=True)
    return on_tr_loader, on_va_loader