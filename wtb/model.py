"""
model.py
========
WTB-DefectNet backbone, implementing all 7 blocks from Proposed_Architecture.docx
at the placements specified there:

    Input 224x224x3
      -> DSPS (stem)                                    56x56x64
      -> Stage 1: TSDB + ASA                             56x56x64
      -> Stage 2: TSDB + ASA + DRFB                      28x28x128
      -> Stage 3: TSDB + ASA + WGFR                      14x14x256
      -> Stage 4: TSDB + ASA + MSCA                        7x7x512
      -> LTCP head                                       9 logits

NOTE on your original wtb_defectnet.py: it only implemented DSPS, TSDB, ASA,
LTCP and used the same plain TSDB+ASA stage at all 4 depths. DRFB, WGFR and
MSCA (three of your seven claimed novel blocks) were not in the code. That
would have made the paper's architecture section and the actual trained
model disagree with each other, which a reviewer (or your advisor) would
flag immediately. This file adds the missing three at the doc's placements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


# ======================================================================
# B1 -- Defect-Scale Pyramid Stem (DSPS)
# ======================================================================
class DSPS(nn.Module):
    """Parallel depthwise dilated convs (d=1,2,3) + Sobel edge branch, fused."""

    def __init__(self, in_ch: int = 3, out_ch: int = 64):
        super().__init__()
        mid = out_ch
        self.b1 = nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, dilation=1)
        self.b2 = nn.Conv2d(in_ch, mid, 3, stride=2, padding=2, dilation=2)
        self.b3 = nn.Conv2d(in_ch, mid, 3, stride=2, padding=3, dilation=3)

        # Sobel edge branch: fixed, non-learnable Gx/Gy kernels applied to a
        # grayscale-luma projection of the RGB input, then downsampled to match.
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = sobel_x.t().clone()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("luma_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.edge_proj = nn.Conv2d(2, mid, 1)

        self.proj = nn.Conv2d(mid * 4, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        # Branches above are stride=2 (224 -> 112). Your plan doc's macro table
        # (section 2) specifies the STEM alone must reach 56x56 before Stage 1
        # (which itself does not downsample). One extra stride-2 pool gives the
        # documented 4x total stem reduction (224 -> 112 -> 56), matching every
        # stage's stated output size in the doc.
        self.pool = nn.MaxPool2d(2, stride=2)

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        gray = (x * self.luma_w).sum(dim=1, keepdim=True)          # (B,1,H,W)
        gx = F.conv2d(gray, self.sobel_x, stride=2, padding=1)
        gy = F.conv2d(gray, self.sobel_y, stride=2, padding=1)
        return torch.cat([gx, gy], dim=1)                          # (B,2,H/2,W/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = self.edge_proj(self._sobel_edges(x))
        feats = torch.cat([self.b1(x), self.b2(x), self.b3(x), edge], dim=1)
        out = self.act(self.bn(self.proj(feats)))
        return self.pool(out)


# ======================================================================
# B2 -- Texture-Structure Disentangling Block (TSDB)
# ======================================================================
class TSDB(nn.Module):
    """LF/HF split via AvgPool residual; two specialized branches; learnable channel gate."""

    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        self.pool = nn.AvgPool2d(k, stride=1, padding=k // 2)
        self.struct = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.GELU(),
        )
        self.texture = nn.Sequential(
            nn.Conv2d(ch, ch, 1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch), nn.BatchNorm2d(ch), nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch * 2, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_low = self.pool(x)
        f_high = x - f_low
        s = self.struct(f_low)
        t = self.texture(f_high)
        g = self.gate(torch.cat([s, t], dim=1))
        return g * t + (1.0 - g) * s


# ======================================================================
# B3 -- Anisotropic Strip Attention (ASA)
# ======================================================================
class ASA(nn.Module):
    """Horizontal + vertical strip pooling -> directional spatial attention."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv_h = nn.Conv1d(ch, ch, 3, padding=1, groups=ch)
        self.conv_v = nn.Conv1d(ch, ch, 3, padding=1, groups=ch)
        self.fuse = nn.Conv2d(ch, ch, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_h = x.mean(dim=3)                     # (B,C,H)
        z_v = x.mean(dim=2)                     # (B,C,W)
        a_h = self.conv_h(z_h).unsqueeze(3)      # (B,C,H,1)
        a_v = self.conv_v(z_v).unsqueeze(2)      # (B,C,1,W)
        a = self.act(self.fuse(a_h + a_v))       # broadcasts to (B,C,H,W)
        return x * a


# ======================================================================
# B4 -- Deformable Receptive Field Block (DRFB)  -- placed at Stage 2
# ======================================================================
class DRFB(nn.Module):
    """
    Offset-predicted grouped deformable convolution (4 groups) inside a
    bottleneck residual, so sampling geometry adapts to UAV perspective
    distortion of the defect shape instead of a rigid grid.
    """

    def __init__(self, ch: int, kernel_size: int = 3, groups: int = 4):
        super().__init__()
        assert ch % groups == 0, "channels must divide evenly by DRFB groups"
        offset_ch = 2 * kernel_size * kernel_size  # offset_groups=1: dx,dy per kernel tap
        self.offset_conv = nn.Conv2d(ch, offset_ch, kernel_size=3, padding=1)
        # zero-init so DRFB starts as an ordinary (non-deformed) conv --
        # standard stabilization trick, avoids garbage offsets early in training
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.deform_conv = DeformConv2d(
            ch, ch, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups
        )
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.GELU()

        mid = max(ch // 2, 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, mid, 1), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, ch, 1), nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        out = self.act(self.bn(self.deform_conv(x, offset)))
        out = self.bottleneck(out)
        return x + out


# ======================================================================
# Haar DWT/IDWT as fixed (non-learnable) grouped convolutions
# ======================================================================
class HaarDWT(nn.Module):
    """
    1-level 2D Haar wavelet transform implemented as a depthwise conv with 4
    fixed, orthonormal filters (LL/LH/HL/HH) per channel. Because the filter
    bank is orthonormal, the inverse transform is exactly the transposed
    convolution with the same weights -- no separate learnable "decoder".
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[1., 1.], [1., 1.]]) * 0.5
        lh = torch.tensor([[1., 1.], [-1., -1.]]) * 0.5
        hl = torch.tensor([[1., -1.], [1., -1.]]) * 0.5
        hh = torch.tensor([[1., -1.], [-1., 1.]]) * 0.5
        bank = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)   # (4,1,2,2)
        weight = bank.repeat(channels, 1, 1, 1)                    # (4C,1,2,2)
        self.register_buffer("weight", weight)

    @staticmethod
    def _pad_to_even(x: torch.Tensor):
        _, _, h, w = x.shape
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (h, w)

    def decompose(self, x: torch.Tensor):
        x, orig_hw = self._pad_to_even(x)
        out = F.conv2d(x, self.weight, stride=2, groups=self.channels)  # (B,4C,H/2,W/2)
        ll = out[:, 0::4]
        lh = out[:, 1::4]
        hl = out[:, 2::4]
        hh = out[:, 3::4]
        return ll, lh, hl, hh, orig_hw

    def reconstruct(self, ll, lh, hl, hh, orig_hw):
        b, c, h2, w2 = ll.shape
        stacked = torch.stack([ll, lh, hl, hh], dim=2).reshape(b, c * 4, h2, w2)
        recon = F.conv_transpose2d(stacked, self.weight, stride=2, groups=self.channels)
        H, W = orig_hw
        return recon[:, :, :H, :W]   # crop off the even-padding, if any


class _ChannelGate(nn.Module):
    """SE-style squeeze-excite gate used inside WGFR to weight sub-bands."""

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ======================================================================
# B5 -- Wavelet-Gated Frequency Refinement (WGFR)  -- placed at Stage 3
# ======================================================================
class WGFR(nn.Module):
    """
    Replaces TSDB's rough AvgPool LF/HF split with an exact, alias-free Haar
    DWT at Stage 3 (the semantic stage where frequency precision matters
    most), refines each sub-band with its own gated conv, and reconstructs
    via IDWT as a targeted residual correction.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.dwt = HaarDWT(ch)
        self.lf_refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch), nn.BatchNorm2d(ch), nn.GELU(),
        )
        self.hf_refine = nn.Sequential(
            nn.Conv2d(ch * 3, ch * 3, 3, padding=1, groups=ch * 3),
            nn.BatchNorm2d(ch * 3), nn.GELU(),
        )
        self.lf_gate = _ChannelGate(ch)
        self.hf_gate = _ChannelGate(ch * 3)
        self.hf_proj = nn.Conv2d(ch * 3, ch * 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh, orig_hw = self.dwt.decompose(x)

        ll = self.lf_gate(self.lf_refine(ll))
        hf = torch.cat([lh, hl, hh], dim=1)
        hf = self.hf_gate(self.hf_refine(hf))
        hf = self.hf_proj(hf)
        lh2, hl2, hh2 = torch.chunk(hf, 3, dim=1)

        recon = self.dwt.reconstruct(ll, lh2, hl2, hh2, orig_hw)
        return x + recon    # targeted frequency correction, residual per doc


# ======================================================================
# B6 -- Multi-Scale Context Aggregator (MSCA)  -- placed at Stage 4
# ======================================================================
class MSCA(nn.Module):
    """Inception-style 1x1 / 3x3-depthwise / 5x5-depthwise parallel branches."""

    def __init__(self, ch: int):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.BatchNorm2d(ch), nn.GELU())
        self.b3 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch), nn.BatchNorm2d(ch), nn.GELU(),
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(ch, ch, 5, padding=2, groups=ch), nn.BatchNorm2d(ch), nn.GELU(),
        )
        self.fuse = nn.Sequential(nn.Conv2d(ch * 3, ch, 1), nn.BatchNorm2d(ch))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)
        out = self.fuse(out)
        return self.act(x + out)


# ======================================================================
# Stage wrapper: downsample -> TSDB (residual) -> ASA -> optional extra block
# ======================================================================
class Stage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True, extra: str = None):
        super().__init__()
        self.down = None
        if downsample or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2 if downsample else 1, padding=1),
                nn.BatchNorm2d(out_ch), nn.GELU(),
            )
        self.tsdb = TSDB(out_ch)
        self.asa = ASA(out_ch)

        assert extra in (None, "drfb", "wgfr", "msca")
        if extra == "drfb":
            self.extra = DRFB(out_ch)
        elif extra == "wgfr":
            self.extra = WGFR(out_ch)
        elif extra == "msca":
            self.extra = MSCA(out_ch)
        else:
            self.extra = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down is not None:
            x = self.down(x)
        x = x + self.tsdb(x)     # residual
        x = self.asa(x)
        if self.extra is not None:
            x = self.extra(x)
        return x


# ======================================================================
# B7 -- Long-Tail Calibrated Prototype Head (LTCP)
# ======================================================================
class LTCP(nn.Module):
    """Cosine-prototype classifier + class-frequency logit adjustment (train-time only)."""

    def __init__(self, feat_dim: int, num_classes: int, tau: float = 16.0, lam: float = 1.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)
        self.tau = tau
        self.lam = lam
        self.register_buffer("log_prior", torch.zeros(num_classes))

    def set_prior(self, class_counts) -> None:
        prior = torch.tensor(class_counts, dtype=torch.float32)
        prior = prior / prior.sum()
        self.log_prior = torch.log(prior + 1e-12).to(self.prototypes.device)

    def forward(self, feat: torch.Tensor, training_adjust: bool = True):
        f = F.normalize(feat, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        cos = f @ p.t()
        logits = self.tau * cos
        if training_adjust and self.training:
            logits = logits + self.lam * self.log_prior.unsqueeze(0)
        return logits, f


# ======================================================================
# Full model
# ======================================================================
class WTBDefectNet(nn.Module):
    def __init__(self, num_classes: int = 9, widths=(64, 128, 256, 512),
                 tau: float = 16.0, lam: float = 1.0):
        super().__init__()
        self.stem = DSPS(3, widths[0])
        self.stage1 = Stage(widths[0], widths[0], downsample=False, extra=None)
        self.stage2 = Stage(widths[0], widths[1], downsample=True, extra="drfb")
        self.stage3 = Stage(widths[1], widths[2], downsample=True, extra="wgfr")
        self.stage4 = Stage(widths[2], widths[3], downsample=True, extra="msca")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = LTCP(widths[3], num_classes, tau=tau, lam=lam)
        self.feat_dim = widths[3]

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        self._last_feat_map = x        # kept around for Grad-CAM later
        feat = self.gap(x).flatten(1)
        logits, f = self.head(feat)
        return logits, f

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
