import math
import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(labels: np.ndarray, max_weight: float = 10.0) -> torch.Tensor:
    """
    Per-task positive class weight = n_neg / n_pos over labelled entries only
    (entries equal to -1 are unlabelled). Clamped so rare tasks do not dominate the loss.
    """
    labels = np.asarray(labels, dtype=float)
    pos = (labels == 1).sum(axis=0)
    neg = (labels == 0).sum(axis=0)
    w = neg / np.maximum(pos, 1)
    return torch.tensor(np.clip(w, 1.0, max_weight), dtype=torch.float32)


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor = None) -> torch.Tensor:
    """Multi-task BCE that ignores unlabelled entries (y < 0)."""
    mask = (y >= 0).float()
    targets = y.clamp(min=0)
    per_ent = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
        pos_weight=None if pos_weight is None else pos_weight.to(logits.device),
    )
    return (per_ent * mask).sum() / mask.sum().clamp(min=1.0)


def warmup_cosine_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.05, min_lr_ratio: float = 0.05):
    """Linear warmup then cosine decay to `min_lr_ratio` of the base learning rate."""
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
