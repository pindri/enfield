import math

import torch


def aps_scores(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-12)
    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=1)

    labels = labels.long()
    # Find the position of each true label in sorted_idx.
    # Create (N,K) where each row compares to the label.
    matches = (sorted_idx == labels[:, None])
    # Sanity check: each row must match exactly once.
    if not torch.all(matches.sum(dim=1) == 1):
        raise RuntimeError("APS: label not found exactly once in sorted indices (bug in label/index alignment).")

    pos = matches.float().argmax(dim=1)  # (N,)
    cum_true = cumsum[torch.arange(probs.size(0)), pos]  # in (0,1]
    score = 1.0 - cum_true  # in [0,1)
    return score.clamp(0.0, 1.0)


def aps_prediction_set_mask(probs_row: torch.Tensor, tau: float) -> torch.Tensor:
    probs_row = probs_row.clamp_min(1e-12)
    sorted_probs, sorted_idx = torch.sort(probs_row, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)

    cumul_target = 1.0 - tau
    # smallest k with cumsum[k] >= cumul_target.
    k = int(torch.searchsorted(cumsum, torch.tensor(cumul_target)).item())
    k = max(k, 0)
    selected = sorted_idx[: k + 1]

    mask = torch.zeros_like(probs_row, dtype=torch.bool)
    mask[selected] = True
    return mask


def split_conformal_threshold(scores: torch.Tensor, alpha: float) -> float:
    """
    Standard split conformal threshold:
      tau = quantile_{k} of scores, where k = ceil((m+1)*(1-alpha))
    Using the "conservative" conformal quantile.
    """
    m = scores.numel()
    k = int(math.ceil((m + 1) * (1 - alpha)))
    k = min(max(k, 1), m)
    # For kth smallest => use torch.kthvalue (1-indexed).
    tau = scores.kthvalue(k).values.item()
    return float(tau)