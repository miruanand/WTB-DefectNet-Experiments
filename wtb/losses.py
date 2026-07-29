"""
losses.py
=========
Composite loss from your plan doc, section 4:

    L_total = L_cb + beta1 * L_LA + beta2 * L_pc

    L_cb  : class-balanced focal loss (effective-number-of-samples weighting)
    L_LA  : logit-adjusted cross-entropy (rare classes get a training-time margin)
    L_pc  : prototype/center loss (pulls features toward their class prototype)
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_weights(counts: List[int], beta: float = 0.999) -> torch.Tensor:
    """
    Effective number of samples weighting (Cui et al., CVPR 2019):
        w_k proportional to (1 - beta) / (1 - beta^n_k)
    Rare classes (e.g. lightning strikes, n≈285) get a much larger weight
    than common ones (e.g. erosion, n≈2799) without the weight blowing up
    the way plain inverse-frequency weighting would.
    """
    counts = np.asarray(counts, dtype=np.float64)
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff_num, 1e-12)
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)


class CompositeLoss(nn.Module):
    def __init__(self, class_counts: List[int], gamma: float = 2.0,
                 beta_la: float = 1.0, beta_pc: float = 0.1, cb_beta: float = 0.999):
        super().__init__()
        self.gamma = gamma
        self.beta_la = beta_la
        self.beta_pc = beta_pc
        self.register_buffer("cb_w", class_balanced_weights(class_counts, cb_beta))

    def forward(self, logits: torch.Tensor, feat: torch.Tensor,
                prototypes: torch.Tensor, target: torch.Tensor):
        # ---- class-balanced focal loss ----
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = -((1 - pt) ** self.gamma) * logpt
        w = self.cb_w.to(logits.device)[target]
        l_cb = (w * focal).mean()

        # ---- logit-adjusted CE ----
        # `logits` already has +lam*log_prior baked in during training (see
        # LTCP.forward), so plain CE here IS the logit-adjusted CE term.
        l_la = F.cross_entropy(logits, target)

        # ---- prototype / center loss ----
        p_y = F.normalize(prototypes[target], dim=1)
        f = F.normalize(feat, dim=1)
        l_pc = (1.0 - (f * p_y).sum(dim=1)).mean()

        total = l_cb + self.beta_la * l_la + self.beta_pc * l_pc
        return total, {
            "loss_cb": l_cb.item(),
            "loss_la": l_la.item(),
            "loss_pc": l_pc.item(),
            "loss_total": total.item(),
        }
