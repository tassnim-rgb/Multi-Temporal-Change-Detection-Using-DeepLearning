"""Change-detection architectures, verbatim from the notebook
(cells 14-28). Includes shared Siamese primitives, FC-Siam-Conc/Diff,
U-Net CD, BIT (Binary Image Transformer), DMFDIL, Bi-UNet Dense,
ConvLSTM-CD, and the image-differencing wrapper used for baselines.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import LEVIR_MEAN, LEVIR_STD


# ── Primitives ───────────────────────────────────────────────────────────────
class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, p=1, s=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(ConvBNReLU(cin, cout), ConvBNReLU(cout, cout))


class EncBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = DoubleConv(cin, cout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x); return s, self.pool(s)


class DecBlock(nn.Module):
    """TransposeConv upsample, concat skip, DoubleConv."""

    def __init__(self, cin, skip_ch, cout):
        super().__init__()
        self.up   = nn.ConvTranspose2d(cin, cin//2, 2, stride=2)
        self.conv = DoubleConv(cin//2 + skip_ch, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


# Shared Siamese Encoder
class SiamEncoder(nn.Module):
    """
    4-level U-Net encoder with weight-shared T1/T2 branches.
    in_ch : 3 (LEVIR) | 13 (ONERA)
    Channel progression: in_ch, c, 2c, 4c, 8c, 16c (bridge)
    """
    def __init__(self, in_ch, base=16):
        super().__init__()
        c = base
        self.e1 = EncBlock(in_ch, c)
        self.e2 = EncBlock(c,   c*2)
        self.e3 = EncBlock(c*2, c*4)
        self.e4 = EncBlock(c*4, c*8)
        self.bridge = DoubleConv(c*8, c*16)

    def forward(self, x):
        s1, p1 = self.e1(x); s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2); s4, p4 = self.e4(p3)
        return [s1, s2, s3, s4], self.bridge(p4)


# ── FC-Siam-Conc ─────────────────────────────────────────────────────────────
class FCSiamConc(nn.Module):
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        c = base
        self.enc = SiamEncoder(in_ch, base)
        # Bridge is cat(b1,b2) -> c*32 channels
        self.d4 = DecBlock(c*32, c*16, c*8)   # in=c*32, skip=cat(s4,s4)=c*16
        self.d3 = DecBlock(c*8,  c*8,  c*4)   # in=c*8,  skip=cat(s3,s3)=c*8
        self.d2 = DecBlock(c*4,  c*4,  c*2)   # in=c*4,  skip=cat(s2,s2)=c*4
        self.d1 = DecBlock(c*2,  c*2,  c)     # in=c*2,  skip=cat(s1,s1)=c*2
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, t1, t2):
        sk1, b1 = self.enc(t1); sk2, b2 = self.enc(t2)
        b = torch.cat([b1, b2], 1)
        d = self.d4(b, torch.cat([sk1[3], sk2[3]], 1))
        d = self.d3(d, torch.cat([sk1[2], sk2[2]], 1))
        d = self.d2(d, torch.cat([sk1[1], sk2[1]], 1))
        d = self.d1(d, torch.cat([sk1[0], sk2[0]], 1))
        return self.head(d)   # (B,1,H,W) logits


# ── FC-Siam-Diff ─────────────────────────────────────────────────────────────
class FCSiamDiff(nn.Module):
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        c = base
        self.enc = SiamEncoder(in_ch, base)
        # Bridge cat(b1,b2) -> c*32; skips are single-branch width
        self.d4 = DecBlock(c*32, c*8, c*8)
        self.d3 = DecBlock(c*8,  c*4, c*4)
        self.d2 = DecBlock(c*4,  c*2, c*2)
        self.d1 = DecBlock(c*2,  c,   c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, t1, t2):
        sk1, b1 = self.enc(t1); sk2, b2 = self.enc(t2)
        b = torch.cat([b1, b2], 1)
        d = self.d4(b, torch.abs(sk1[3]-sk2[3]))
        d = self.d3(d, torch.abs(sk1[2]-sk2[2]))
        d = self.d2(d, torch.abs(sk1[1]-sk2[1]))
        d = self.d1(d, torch.abs(sk1[0]-sk2[0]))
        return self.head(d)


# ── Attention modules + U-Net CD ─────────────────────────────────────────────
class ChangeAttn(nn.Module):
    """SE channel-attention + CBAM spatial-attention on difference maps."""

    def __init__(self, ch):
        super().__init__()
        r = max(ch//4, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, r), nn.ReLU(inplace=True),
            nn.Linear(r, ch), nn.Sigmoid())
        self.sp = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, d):
        B, C, H, W = d.shape
        d = d * self.se(d).view(B, C, 1, 1)
        sp = self.sp(torch.cat([d.mean(1, keepdim=True),
                                d.max(1, keepdim=True).values], 1))
        return d * sp


class UNetCD(nn.Module):
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        c = base; fuse = in_ch*2
        # FC-EF encoder
        self.e1 = EncBlock(fuse, c); self.e2 = EncBlock(c, c*2)
        self.e3 = EncBlock(c*2, c*4); self.e4 = EncBlock(c*4, c*8)
        self.br = DoubleConv(c*8, c*16)
        # Auxiliary Siamese (shared weights) for diff skips
        self.siam = SiamEncoder(in_ch, base)
        # Change attention gates
        self.ca4 = ChangeAttn(c*8); self.ca3 = ChangeAttn(c*4)
        self.ca2 = ChangeAttn(c*2); self.ca1 = ChangeAttn(c)
        # Decoder
        self.d4 = DecBlock(c*16, c*8, c*8)
        self.d3 = DecBlock(c*8,  c*4, c*4)
        self.d2 = DecBlock(c*4,  c*2, c*2)
        self.d1 = DecBlock(c*2,  c,   c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, t1, t2):
        x = torch.cat([t1, t2], 1)
        s1, p1 = self.e1(x); s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2); s4, p4 = self.e4(p3)
        b = self.br(p4)
        sk1, _ = self.siam(t1); sk2, _ = self.siam(t2)
        d4 = self.ca4(torch.abs(sk1[3]-sk2[3]))
        d3 = self.ca3(torch.abs(sk1[2]-sk2[2]))
        d2 = self.ca2(torch.abs(sk1[1]-sk2[1]))
        d1 = self.ca1(torch.abs(sk1[0]-sk2[0]))
        d = self.d4(b, d4); d = self.d3(d, d3)
        d = self.d2(d, d2); d = self.d1(d, d1)
        return self.head(d)


# ── BIT (Binary Image Transformer) ───────────────────────────────────────────
class TransformerLayer(nn.Module):
    """Single Transformer encoder layer (pre-norm)."""

    def __init__(self, dim, n_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*mlp_ratio, dim), nn.Dropout(dropout))

    def forward(self, x):
        x2 = self.norm1(x)
        x  = x + self.attn(x2, x2, x2)[0]
        x  = x + self.mlp(self.norm2(x))
        return x


class BIT(nn.Module):
    """
    Binary Image Transformer for Change Detection.
    in_ch   : 3 (LEVIR)
    base    : channel multiplier
    n_tok   : number of learnable change tokens in the decoder
    n_layers: depth of transformer encoder and decoder
    """
    def __init__(self, in_ch=3, base=16, n_tok=4, n_layers=2):
        super().__init__()
        c = base
        # Siamese CNN encoder (shared)
        self.enc = SiamEncoder(in_ch, base)
        # Token embedding: project bridge features to transformer dim
        self.tok_dim = c * 16
        # Transformer encoder: contextualise both feature sets
        self.tr_enc = nn.Sequential(
            *[TransformerLayer(self.tok_dim) for _ in range(n_layers)])
        # Learnable change tokens (query) for transformer decoder
        self.change_tokens = nn.Parameter(
            torch.randn(1, n_tok, self.tok_dim))
        # Transformer decoder: cross-attend change tokens to encoded features
        self.tr_dec_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.tok_dim, nhead=8,
                dim_feedforward=self.tok_dim*4,
                dropout=0.1, batch_first=True)
            for _ in range(n_layers)])
        # Convolutional decoder
        self.d4 = DecBlock(c*32, c*8, c*8)
        self.d3 = DecBlock(c*8,  c*4, c*4)
        self.d2 = DecBlock(c*4,  c*2, c*2)
        self.d1 = DecBlock(c*2,  c,   c)
        # Project token context back into spatial feature
        self.tok2feat = nn.Linear(self.tok_dim, self.tok_dim)
        self.head     = nn.Conv2d(c, 1, 1)

    def forward(self, t1, t2):
        sk1, b1 = self.enc(t1)
        sk2, b2 = self.enc(t2)

        # Tokenise bridge features: (B,C,H,W) -> (B, H*W, C)
        B, C, H, W = b1.shape
        tok1 = b1.flatten(2).permute(0, 2, 1)   # (B, HW, C)
        tok2 = b2.flatten(2).permute(0, 2, 1)
        tokens = torch.cat([tok1, tok2], 1)      # (B, 2*HW, C)

        # Transformer encoder: global context
        mem = self.tr_enc(tokens)                # (B, 2*HW, C)

        # Transformer decoder: change tokens attend to memory
        q = self.change_tokens.expand(B, -1, -1) # (B, n_tok, C)
        for layer in self.tr_dec_layers:
            q = layer(q, mem)

        # Project decoded tokens and broadcast back into spatial
        g = self.tok2feat(q.mean(1))             # (B, C)
        g = g.view(B, C, 1, 1)                   # (B, C, 1, 1)

        # Convolutional decoder with Siamese diff skips + global modulation
        b = torch.cat([b1*g, b2*g], 1)           # modulated bridge
        d = self.d4(b,  torch.abs(sk1[3]-sk2[3]))
        d = self.d3(d,  torch.abs(sk1[2]-sk2[2]))
        d = self.d2(d,  torch.abs(sk1[1]-sk2[1]))
        d = self.d1(d,  torch.abs(sk1[0]-sk2[0]))
        return self.head(d)


# ── DMFDIL (multi-scale diff + intra-class loss) ─────────────────────────────
class FPN_Neck(nn.Module):
    """Top-down FPN that fuses multi-scale difference features."""
    def __init__(self, channels):
        # channels: list [c, c*2, c*4, c*8] bottom-up
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Conv2d(ch, channels[0], 1) for ch in channels])
        self.smooth = nn.ModuleList([
            ConvBNReLU(channels[0], channels[0]) for _ in channels])

    def forward(self, feats):
        # feats: [d1, d2, d3, d4]  (smallest to largest spatial)
        lats = [l(f) for l, f in zip(self.lat, feats)]
        out  = lats[-1]
        results = [out]
        for lat in reversed(lats[:-1]):
            out = F.interpolate(out, size=lat.shape[2:],
                                mode='bilinear', align_corners=False) + lat
            results.insert(0, out)
        return [s(r) for s, r in zip(self.smooth, results)]


class DMFDIL(nn.Module):
    """
    Deep Multi-scale Feature Difference with Intra-class Loss.
    in_ch : 3 (LEVIR)
    """
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        c = base
        self.enc = SiamEncoder(in_ch, base)
        # FPN neck over difference pyramid
        self.fpn = FPN_Neck([c, c*2, c*4, c*8])
        # Decoder consumes FPN output (all at width c); fuse 4 upsampled levels
        self.fuse = DoubleConv(c*4, c*2)   # cat 4 levels at c each
        self.head = nn.Conv2d(c*2, 1, 1)
        from .losses import IntraClassLoss
        self.icl = IntraClassLoss()

    def forward(self, t1, t2, mask=None):
        sk1, b1 = self.enc(t1); sk2, b2 = self.enc(t2)
        # Multi-scale difference maps
        diffs = [torch.abs(sk1[i]-sk2[i]) for i in range(4)]   # 4 scales
        # FPN fusion
        fpn_outs = self.fpn(diffs)   # all at width c, varying spatial
        # Upsample all to scale of fpn_outs[0] (largest spatial)
        target_sz = fpn_outs[0].shape[2:]
        fused = torch.cat([
            F.interpolate(f, size=target_sz,
                          mode='bilinear', align_corners=False)
            for f in fpn_outs], 1)              # (B, c*4, H, W)
        out = self.head(self.fuse(fused))       # (B, 1, H, W)
        if mask is not None and self.training:
            # Return logits + intra-class loss for training
            return out, self.icl(fused, mask)
        return out


# ── Bi-UNet Dense (deep supervision) ─────────────────────────────────────────
class BiUNetDense(nn.Module):
    """
    Bi-temporal Dense U-Net with deep supervision.
    Channel widths are auto-detected from SiamEncoder via a dry forward pass,
    so this class is robust to any base width or encoder implementation.
    """
    def __init__(self, in_ch=13, base=16):
        super().__init__()
        self.enc = SiamEncoder(in_ch, base)

        # Auto-detect skip and bridge channel widths
        with torch.no_grad():
            _dummy = torch.zeros(1, in_ch, 64, 64)
            _sk, _b = self.enc(_dummy)
            sk_ch = [s.shape[1] for s in _sk]  # [c0, c1, c2, c3], fine->coarse
            br_ch = _b.shape[1] * 2            # bridge = cat(b1, b2)

        d4_out = sk_ch[3] // 2
        d3_out = sk_ch[2] // 2
        d2_out = sk_ch[1] // 2
        d1_out = sk_ch[0] // 2

        # Decoder blocks
        self.d4 = DecBlock(br_ch,  sk_ch[3], d4_out)
        self.d3 = DecBlock(d4_out, sk_ch[2], d3_out)
        self.d2_conv = DoubleConv(d3_out + sk_ch[1], d2_out)
        self.d1_conv = DoubleConv(d2_out + sk_ch[0], d1_out)

        # Deep supervision heads
        self.aux4 = nn.Conv2d(d4_out, 1, 1)
        self.aux3 = nn.Conv2d(d3_out, 1, 1)
        self.aux2 = nn.Conv2d(d2_out, 1, 1)
        self.aux1 = nn.Conv2d(d1_out, 1, 1)

    def _up(self, x, target):
        """Upsample x to match target's spatial dimensions if needed."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(
                x, size=target.shape[2:], mode='bilinear', align_corners=False
            )
        return x

    def forward(self, t1, t2):
        sk1, b1 = self.enc(t1)
        sk2, b2 = self.enc(t2)

        # Absolute difference at each encoder scale
        diff = [torch.abs(sk1[i] - sk2[i]) for i in range(4)]

        bridge = torch.cat([b1, b2], dim=1)

        # Decoder
        d4 = self.d4(bridge, diff[3])
        a4 = self.aux4(d4)

        d3 = self.d3(d4, diff[2])
        a3 = self.aux3(d3)

        d2 = self.d2_conv(torch.cat([self._up(d3, diff[1]), diff[1]], dim=1))
        a2 = self.aux2(d2)

        d1 = self.d1_conv(torch.cat([self._up(d2, diff[0]), diff[0]], dim=1))
        a1 = self.aux1(d1)

        if self.training:
            return a1, [a2, a3, a4]
        return a1


# ── ConvLSTM-CD ──────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell. input_dim and hidden_dim are channel counts."""

    def __init__(self, input_dim, hidden_dim, kernel=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel // 2
        # Gates: input, forget, output, cell, all in one conv
        self.gates = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim * 4,
            kernel, padding=pad, bias=True)

    def forward(self, x, h, c):
        # x : (B,C,H,W) | h,c : (B,hidden,H,W)
        combined = torch.cat([x, h], 1)
        gates    = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        o = torch.sigmoid(o); g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, B, H, W, device):
        return (torch.zeros(B, self.hidden_dim, H, W, device=device),
                torch.zeros(B, self.hidden_dim, H, W, device=device))


class ConvLSTMCD(nn.Module):
    """
    ConvLSTM Change Detection for multispectral data.
    in_ch : 13 (ONERA)
    The ConvLSTM processes the sequence [bridge_T1, bridge_T2]
    and uses the final hidden state as the temporal fusion feature.
    """

    def __init__(self, in_ch=13, base=16):
        super().__init__()
        c = base
        self.enc  = SiamEncoder(in_ch, base)
        self.lstm = ConvLSTMCell(c*16, c*16)
        # Decoder with LSTM hidden state as bridge
        self.d4 = DecBlock(c*16, c*8, c*8)
        self.d3 = DecBlock(c*8,  c*4, c*4)
        self.d2 = DecBlock(c*4,  c*2, c*2)
        self.d1 = DecBlock(c*2,  c,   c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, t1, t2):
        sk1, b1 = self.enc(t1)
        sk2, b2 = self.enc(t2)

        B, C, H, W = b1.shape
        h, c = self.lstm.init_hidden(B, H, W, b1.device)

        # Temporal sequence: T1, T2
        h, c = self.lstm(b1, h, c)
        h, c = self.lstm(b2, h, c)

        # Decode from LSTM hidden state (temporal fusion)
        d = self.d4(h,  torch.abs(sk1[3]-sk2[3]))
        d = self.d3(d,  torch.abs(sk1[2]-sk2[2]))
        d = self.d2(d,  torch.abs(sk1[1]-sk2[1]))
        d = self.d1(d,  torch.abs(sk1[0]-sk2[0]))
        return self.head(d)


# ── Image-differencing wrapper (baselines) ───────────────────────────────────
class ImgDiffModel(nn.Module):
    """Thin wrapper so image differencing plugs into the same eval code."""

    def __init__(self, is_levir):
        super().__init__()
        self.is_levir = is_levir
        self.dummy = nn.Parameter(torch.zeros(1))   # keeps nn.Module happy

    def forward(self, i1, i2):
        from sklearn.decomposition import PCA
        import cv2 as _cv2
        MEAN = np.array(LEVIR_MEAN, np.float32)
        STD  = np.array(LEVIR_STD,  np.float32)
        out = []
        for b in range(i1.shape[0]):
            if self.is_levir:
                a = i1[b].cpu().permute(1,2,0).numpy()*STD+MEAN
                c = i2[b].cpu().permute(1,2,0).numpy()*STD+MEAN
                diff = np.abs(c-a).mean(-1).astype(np.float32)
            else:
                dv = (i2[b]-i1[b]).cpu().permute(1,2,0).numpy()
                H, W, C = dv.shape
                diff = np.abs(
                    PCA(1).fit_transform(dv.reshape(-1, C))
                ).reshape(H, W).astype(np.float32)
            # Otsu, return as float logit-like in [0,1]
            norm = _cv2.normalize(diff, None, 0, 1, _cv2.NORM_MINMAX)
            out.append(torch.from_numpy(norm))
        return torch.stack(out).unsqueeze(1)


def parameter_summary():
    """Print per-model parameter counts (mirrors the notebook's summary)."""
    for name, cls, ich in [
        ('FC-Siam-Conc',  FCSiamConc,  3),
        ('FC-Siam-Diff',  FCSiamDiff,  3),
        ('U-Net CD',      UNetCD,      3),
        ('BIT',           BIT,         3),
        ('DMFDIL',        DMFDIL,      3),
        ('Bi-UNet Dense', BiUNetDense, 13),
        ('ConvLSTM-CD',   ConvLSTMCD,  13),
    ]:
        n = sum(p.numel() for p in cls(ich).parameters())
        print(f'  {name:<18}: {n/1e6:.2f} M')