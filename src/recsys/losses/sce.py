"""Scalable CE: projection buckets, hard subsets, maximum loss across buckets.

Adapted from AIRI-Institute/Scalable-SASRec (MIT); see THIRD_PARTY.md.
Padding is removed before selection; only active catalog embeddings are passed in.
"""

import torch
from torch.nn import functional as F


def scalable_cross_entropy(
    x,
    targets,
    weight,
    n_buckets=8,
    bucket_size_x=64,
    bucket_size_y=256,
    mix_x=False,
    projections=None,
):
    if x.ndim != 2 or not len(x):
        raise ValueError("SCE needs a nonempty [positions, hidden] tensor")
    if min(n_buckets, bucket_size_x, bucket_size_y) < 1:
        raise ValueError("SCE bucket counts/sizes must be positive")
    bx, by = min(bucket_size_x, len(x)), min(bucket_size_y, len(weight))
    with torch.no_grad():
        if projections is not None:
            buckets = projections.to(x)
            n_buckets = len(buckets)
        elif mix_x:
            omega = torch.randn(len(x), n_buckets, device=x.device, dtype=x.dtype)
            buckets = omega.T @ x / x.shape[1] ** 0.25
        else:
            buckets = (
                torch.randn(n_buckets, x.shape[1], device=x.device, dtype=x.dtype)
                / x.shape[1] ** 0.25
            )
        selected_x = (buckets @ x.T).topk(bx, dim=1).indices
        selected_y = (buckets @ weight.T).topk(by, dim=1).indices
    logits = x[selected_x] @ weight[selected_y].transpose(1, 2)
    logits = logits.masked_fill(targets[selected_x, None] == selected_y[:, None, :], -torch.inf)
    positive = (x * weight[targets]).sum(-1)
    logits = torch.cat([logits, positive[selected_x].unsqueeze(-1)], dim=-1)
    losses = F.cross_entropy(
        logits.flatten(0, 1),
        torch.full((n_buckets * bx,), by, device=x.device, dtype=torch.long),
        reduction="none",
    )
    reduced = x.new_zeros(len(x)).scatter_reduce(
        0, selected_x.flatten(), losses, reduce="amax", include_self=False
    )
    covered = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    covered[selected_x.flatten()] = True
    # Include genuinely zero losses; an uncovered position is not the same as zero CE.
    return reduced[covered].mean()
