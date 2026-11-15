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


def _largest_module_list(module: torch.nn.Module) -> torch.nn.ModuleList:
    """The Transformer block stack of a HF model (e.g. MoLFormer's `encoder.layer`)."""
    lists = [m for m in module.modules() if isinstance(m, torch.nn.ModuleList)]
    if not lists:
        raise ValueError("no ModuleList found in encoder")
    return max(lists, key=len)


def configure_encoder_finetuning(graph_encoder, smiles_encoder, image_encoder, spectrum_encoder,
                                 enabled: bool = True, smiles_top_blocks: int = 2):
    """
    Freeze all encoder weights, then (if enabled) unfreeze the parts worth adapting to the
    toxicity task: the whole graph encoder, the top `smiles_top_blocks` MoLFormer blocks plus
    its MLP head, ResNet stage 4 plus the image MLP head, and the spectrum MLP head. The
    pretrained CReSS NMR model always stays frozen. Returns the list of unfrozen encoder
    parameters so they can be given a lower learning rate.
    """
    for net in (graph_encoder, smiles_encoder, image_encoder, spectrum_encoder):
        for p in net.parameters():
            p.requires_grad = False
    spectrum_encoder.missing_token.requires_grad = True

    if not enabled:
        return []

    params = list(graph_encoder.parameters())
    params += list(smiles_encoder.proj.parameters())
    for block in _largest_module_list(smiles_encoder.encoder)[-smiles_top_blocks:]:
        params += list(block.parameters())
    params += list(image_encoder.backbone.layer4.parameters())
    params += list(image_encoder.mlp.parameters())
    params += list(spectrum_encoder.fc1.parameters()) + list(spectrum_encoder.fc2.parameters())
    for p in params:
        p.requires_grad = True
    return params


def snapshot_encoders(*encoders):
    return [{k: v.detach().clone() for k, v in e.state_dict().items()} for e in encoders]


def restore_encoders(encoders, snapshots):
    """Reset shared encoders so every Optuna trial / ensemble seed starts from the same weights."""
    for e, s in zip(encoders, snapshots):
        e.load_state_dict(s, strict=False)
