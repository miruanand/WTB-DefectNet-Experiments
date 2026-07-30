"""
model.py
========
WTB-DefectNet backbone, implementing all 7 blocks from Proposed_Architecture.docx
at the placements specified there:

    Input 224x224x3
      -> DSPS (stem)                                    56x56x64
      -> Stage 1: TSDB + ASA                             56x56x64
      -> Transition Layer
      -> Stage 2: TSDB + ASA + DRFB                      28x28x128
      -> Transition Layer
      -> Stage 3: TSDB + ASA + WGFR                      14x14x256
      -> Transition Layer
      -> Stage 4: TSDB + ASA + MSCA                        7x7x512
      -> LTCP head                                       9 logits

CHANGELOG vs Try_2 (Try_2 -> Try_3): Try_2 scored WORSE than exp1 on every
class but one (macro-F1 0.6537 -> 0.5924, test set) despite train_acc
staying at ~74%, i.e. the gap that opened up was a genuine backbone
capacity problem introduced by the doc-alignment edits, not the EMA
prototypes (lightning-strikes recall was fine in both runs). Two real bugs
found and fixed:

  1. DSPS (B1) rank-3 bottleneck. `depthwise_dilated()` used
     `nn.Conv2d(in_ch, in_ch, ...)` with groups=in_ch=3 -- i.e. exactly ONE
     filter per input colour channel per dilation rate. That means each
     dilated branch could only ever produce 3 linearly-independent spatial
     response maps, no matter how wide the pointwise projection after it
     was (a linear 1x1 conv can rescale/recombine those 3 maps into 64
     channels, but can't invent new spatial patterns). This is a real rank
     bottleneck, not a stylistic issue -- it's the main reason Try_2's stem
     was strictly weaker than a standard-conv stem. Fixed by giving each
     branch a depthwise multiplier (`depth_multiplier=8` below, so 8
     independent filters per input channel = 24 spatial patterns per
     branch instead of 3). Still genuinely depthwise (groups=in_ch), still
     what the doc's Novelty section calls for ("parallel depthwise dilated
     branches") -- just not artificially starved of filters. Also
     restructured to concat all 4 branches THEN apply a single 1x1
     pointwise conv, matching the doc's literal component order
     ("Feature Concatenation" -> "1x1 Pointwise Convolution") instead of
     Try_2's per-branch-then-final double pointwise.

  2. Transition Layer AvgPool blur (tried in Try_3, REVERTED here). The
     doc lists the downsample step as "2x2 Average Pooling (OR Strided
     Convolution)". Try_3 swapped AvgPool for a randomly-initialized
     learned stride-2 depthwise conv at all 3 stage transitions. Result:
     val_macro_f1 stopped climbing and oscillated hard (0.44/0.35/0.47/
     0.39/...) instead of trending up like exp1 did over the same epochs,
     and early stopping triggered at epoch 43 with best=0.4766 -- worse
     than both exp1 and Try_2 at that point. AvgPool is parameter-free and
     behaves identically from epoch 1; 3 randomly-initialized learnable
     downsamplers stacked in series have to individually find their way to
     something useful before they stop actively distorting the feature
     maps, which is a much more plausible explanation for the instability
     than the DSPS change (which only touches the stem, once). REVERTED
     back to AvgPool for this run -- still one of the doc's two listed
     options, and the one with an actual stable track record on this
     dataset. The learned-downsample idea isn't necessarily wrong, but it
     needs a gentler entry (e.g. average-kernel initialization) to be
     tested properly, which is future work, not this run.

  3. Added a light dropout (`head_dropout`, default 0.15, see config.py)
     on the pooled feature vector right before LTCP. The capacity fix in
     (1) makes the backbone strictly more expressive, which raises
     overfitting risk on a 9-class, ~4.5k-image train set; this keeps
     train/val from diverging again without limiting what the corrected
     stem can represent.

  4. `patience` raised 20 -> 35 (config.py). exp1's own best epoch was
     120 (the very last one) and it was still improving at epoch 43
     (0.578 and climbing) -- a 20-epoch patience window is short relative
     to how slowly this model's val_macro_f1 trends upward, and it cut
     Try_3 off well before it could recover from a noisy patch.

TSDB, ASA, DRFB, WGFR, MSCA, and LTCP itself are unchanged from Try_2 --
the per-class breakdown didn't point at them (rare-class recall was
already fine), so they were left alone rather than changed speculatively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


# ======================================================================
# B1 -- Defect-Scale Pyramid Stem (DSPS)
# ======================================================================
class DSPS(nn.Module):
    """
    Parallel DEPTHWISE dilated convs (d=1,2,3) + Sobel edge branch,
    concatenated and fused by a single 1x1 pointwise conv, plus a
    residual-fusion shortcut from the raw input.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 64, depth_multiplier: int = 8):
        super().__init__()
        # `depth_multiplier` filters PER INPUT CHANNEL per dilated branch.
        # On a 3-channel RGB input, depth_multiplier=1 (groups=in_ch,
        # out_channels=in_ch) gives each branch exactly one filter per
        # colour channel -- only 3 linearly-independent spatial patterns,
        # regardless of how wide any pointwise conv placed after it is.
        # depth_multiplier=8 gives 24 independent filters per branch
        # instead, fixing that rank bottleneck while staying genuinely
        # depthwise (groups=in_ch) as the doc's Novelty section specifies.
        branch_ch = in_ch * depth_multiplier

        def depthwise_dilated(dilation: int) -> nn.Conv2d:
            return nn.Conv2d(in_ch, branch_ch, 3, stride=2, padding=dilation,
                              dilation=dilation, groups=in_ch)

        self.b1 = depthwise_dilated(1)
        self.b2 = depthwise_dilated(2)
        self.b3 = depthwise_dilated(3)

        # Sobel edge branch: fixed, non-learnable Gx/Gy kernels applied to a
        # grayscale-luma projection of the RGB input, then downsampled to match.
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = sobel_x.t().clone()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("luma_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.edge_proj = nn.Conv2d(2, branch_ch, 1)

        # Doc order: "Feature Concatenation" -> "1x1 Pointwise Convolution"
        # -- concat all 4 raw branches first, THEN a single pointwise fuse
        # (Try_2 projected each branch individually before concatenating,
        # then fused again -- a redundant double pointwise that didn't
        # match the doc and didn't help capacity either).
        concat_ch = branch_ch * 4
        self.proj = nn.Conv2d(concat_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        # Branches above are stride=2 (224 -> 112). One extra stride-2 pool
        # gives the documented 4x total stem reduction (224 -> 112 -> 56).
        self.pool = nn.MaxPool2d(2, stride=2)

        # Residual Fusion (doc-listed DSPS component). A cheap 1x1,
        # stride-4 shortcut from the raw input straight to the stem output,
        # matching the stem's total downsampling factor (224 -> 56).
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=4)

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        gray = (x * self.luma_w).sum(dim=1, keepdim=True)          # (B,1,H,W)
        gx = F.conv2d(gray, self.sobel_x, stride=2, padding=1)
        gy = F.conv2d(gray, self.sobel_y, stride=2, padding=1)
        return torch.cat([gx, gy], dim=1)                          # (B,2,H/2,W/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = self.edge_proj(self._sobel_edges(x))
        feats = torch.cat([self.b1(x), self.b2(x), self.b3(x), edge], dim=1)
        out = self.act(self.bn(self.proj(feats)))
        out = self.pool(out)
        out = out + self.residual(x)     # residual fusion
        return out


# ======================================================================
# B2 -- Texture-Structure Disentangling Block (TSDB)  -- unchanged
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
# B3 -- Anisotropic Strip Attention (ASA)  -- unchanged
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
# B4 -- Deformable Receptive Field Block (DRFB)  -- Stage 2, unchanged
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
        offset_ch = 2 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(ch, offset_ch, kernel_size=3, padding=1)
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
# Haar DWT/IDWT as fixed (non-learnable) grouped convolutions -- unchanged
# ======================================================================
class HaarDWT(nn.Module):
    """
    1-level 2D Haar wavelet transform implemented as a depthwise conv with 4
    fixed, orthonormal filters (LL/LH/HL/HH) per channel.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[1., 1.], [1., 1.]]) * 0.5
        lh = torch.tensor([[1., 1.], [-1., -1.]]) * 0.5
        hl = torch.tensor([[1., -1.], [1., -1.]]) * 0.5
        hh = torch.tensor([[1., -1.], [-1., 1.]]) * 0.5
        bank = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        weight = bank.repeat(channels, 1, 1, 1)
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
        out = F.conv2d(x, self.weight, stride=2, groups=self.channels)
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
        return recon[:, :, :H, :W]


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
# B5 -- Wavelet-Gated Frequency Refinement (WGFR)  -- Stage 3, unchanged
# ======================================================================
class WGFR(nn.Module):
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
        return x + recon


# ======================================================================
# B6 -- Multi-Scale Context Aggregator (MSCA)  -- Stage 4, unchanged
# ======================================================================
class MSCA(nn.Module):
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
# Transition Layer (doc-specified, between stages)
# ======================================================================
class TransitionLayer(nn.Module):
    """
    1x1 Conv -> BN -> GELU -> 2x2 AvgPool, as listed in the doc's
    "Transition Layer" section: "2x2 Average Pooling (or Strided
    Convolution)". Used between Stage1->2, Stage2->3, Stage3->4.

    Try_3 swapped this for a learned stride-2 depthwise conv (the doc's
    other listed option). That run's val_macro_f1 oscillated hard instead
    of trending up and early-stopped at a worse point than either exp1 or
    Try_2 -- most likely because 3 randomly-initialized learnable
    downsamplers in series have to each learn their way to something
    useful before they stop distorting the feature maps, unlike AvgPool
    which is parameter-free and behaves identically from epoch 1. Reverted
    to AvgPool here since it has an actual stable track record on this
    dataset; the learned-downsample idea would need a gentler entry (e.g.
    average-kernel initialization) to be worth re-testing.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.proj(x))))


# ======================================================================
# Stage wrapper: [transition] -> TSDB (residual) -> ASA -> optional extra block
# ======================================================================
class Stage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True, extra: str = None):
        super().__init__()
        self.down = None
        if downsample:
            self.down = TransitionLayer(in_ch, out_ch)
        elif in_ch != out_ch:
            # channel-only projection, no spatial downsampling (not
            # currently exercised anywhere in WTBDefectNet, kept for safety)
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.GELU(),
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
    """
    Cosine-prototype classifier + class-frequency logit adjustment
    (train-time only) + an EMA-updated prototype memory bank.

    Prototypes are now a registered buffer, NOT an nn.Parameter -- they are
    never updated by backprop/the optimizer. Instead, update_prototypes()
    must be called once per training step (after optimizer.step()) with the
    batch's backbone features and labels; it EMA-updates only the class
    prototypes that were actually present in that batch, leaving the rest
    untouched. This is what lets a rare class like lightning strikes keep a
    stable running estimate across steps, instead of only updating (noisily,
    via gradient) on the rare steps where it happens to appear.
    """

    def __init__(self, feat_dim: int, num_classes: int, tau: float = 16.0,
                 lam: float = 1.0, momentum: float = 0.9):
        super().__init__()
        self.register_buffer("prototypes", torch.randn(num_classes, feat_dim) * 0.01)
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.tau = tau
        self.lam = lam
        self.momentum = momentum
        self.num_classes = num_classes
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

    @torch.no_grad()
    def update_prototypes(self, feat: torch.Tensor, labels: torch.Tensor) -> None:
        """
        EMA-update the prototype memory bank from one batch of backbone
        features. Call this AFTER optimizer.step() each training step, e.g.:

            logits, feat = model(imgs)
            loss.backward(); optimizer.step()
            model.head.update_prototypes(feat.detach(), labels)

        Classes absent from this batch are left untouched (their EMA
        estimate simply carries forward from the last batch they appeared
        in) -- this is exactly the "lightning-strike samples appearing once
        or twice per mini-batch" scenario the architecture doc calls out.
        """
        f = F.normalize(feat, dim=1)
        for c in labels.unique():
            c = int(c.item())
            class_feat = f[labels == c]
            if class_feat.numel() == 0:
                continue
            class_mean = F.normalize(class_feat.mean(dim=0), dim=0)
            if not bool(self.initialized[c]):
                self.prototypes[c] = class_mean
                self.initialized[c] = True
            else:
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c] + (1.0 - self.momentum) * class_mean
                )


# ======================================================================
# Full model
# ======================================================================
class WTBDefectNet(nn.Module):
    def __init__(self, num_classes: int = 9, widths=(64, 128, 256, 512),
                 tau: float = 16.0, lam: float = 1.0, proto_momentum: float = 0.9,
                 head_dropout: float = 0.15):
        super().__init__()
        self.stem = DSPS(3, widths[0])
        self.stage1 = Stage(widths[0], widths[0], downsample=False, extra=None)
        self.stage2 = Stage(widths[0], widths[1], downsample=True, extra="drfb")
        self.stage3 = Stage(widths[1], widths[2], downsample=True, extra="wgfr")
        self.stage4 = Stage(widths[2], widths[3], downsample=True, extra="msca")
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Light dropout on the pooled feature vector, before LTCP. Not a
        # doc component -- added because the DSPS/Transition fixes above
        # make the backbone strictly more expressive than Try_2, which
        # raises overfitting risk on a train_acc-vs-val_macro_f1 gap that
        # was already present (~74% vs ~55-58%). Dropout only, no other
        # regularization changes, so it's easy to dial back (head_dropout=0)
        # if the next run underfits instead.
        self.dropout = nn.Dropout(p=head_dropout)
        self.head = LTCP(widths[3], num_classes, tau=tau, lam=lam, momentum=proto_momentum)
        self.feat_dim = widths[3]

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        self._last_feat_map = x        # kept around for Grad-CAM
        feat = self.gap(x).flatten(1)
        # Dropout only regularizes the LOGITS pathway. `feat` itself (raw,
        # no dropout) is what train.py hands to the composite loss's
        # prototype/center term and to update_prototypes() for the EMA
        # bank -- those exist specifically to stabilize rare-class (e.g.
        # lightning-strike) representations, so they should see clean
        # features rather than a randomly-zeroed copy of them.
        logits, _ = self.head(self.dropout(feat))
        return logits, feat

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
