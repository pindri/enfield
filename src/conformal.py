import math

import torch
from matplotlib import pyplot as plt


# -----------------------------
# Conformal scores and stuff.
# -----------------------------

def aps_scores(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-12)
    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=1)

    matches = (sorted_idx == labels[:, None])
    pos = matches.float().argmax(dim=1)
    cum_true = cumsum[torch.arange(probs.size(0)), pos]
    return cum_true.clamp(0.0, 1.0)


def aps_prediction_set_mask(probs_row: torch.Tensor, tau: float) -> torch.Tensor:
    probs_row = probs_row.clamp_min(1e-12)
    sorted_probs, sorted_idx = torch.sort(probs_row, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)

    k = int(torch.searchsorted(cumsum, torch.tensor(tau)).item())
    k = min(max(k, 0), len(probs_row) - 1)

    selected = sorted_idx[: k + 1]
    mask = torch.zeros_like(probs_row, dtype=torch.bool)
    mask[selected] = True
    return mask


def split_conformal_threshold(scores: torch.Tensor, alpha: float) -> float:
    """
    Standard split conformal threshold:
      tau = quantile_{k} of scores, where k = ceil((m+1)*(1-alpha))
    This could be quite conservative but should work.
    """
    m = scores.numel()
    k = int(math.ceil((m + 1) * (1 - alpha)))
    k = min(max(k, 1), m)
    # For kth smallest => use torch.kthvalue (1-indexed).
    tau = scores.kthvalue(k).values.item()
    return float(tau)


# -----------------------------
# Diagnostics.
# -----------------------------

def summarize_scores(name, scores):
    s = scores.detach().cpu().float()
    qs = torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    vals = torch.quantile(s, qs)
    print(
        f"[{name}] "
        f"min={vals[0]:.4f} "
        f"median={vals[1]:.4f} "
        f"p90={vals[2]:.4f} "
        f"p95={vals[3]:.4f} "
        f"p99={vals[4]:.4f} "
        f"max={vals[5]:.4f}"
    )


def summarize_set_sizes(name, probs):
   p = probs.detach().cpu().float()
   print(
       f"[{name}] "
       f"min_prob_sum={float(p.sum(dim=1).min()):.4f} "
       f"max_prob_sum={float(p.sum(dim=1).max()):.4f} "
       f"min_maxprob={float(p.max(dim=1).values.min()):.4f}"
   )


def save_score_hist(scores, path, title):
    s = scores.detach().cpu().numpy()
    plt.figure(figsize=(6,4))
    plt.hist(s, bins=50)
    plt.xlabel("APS score")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()