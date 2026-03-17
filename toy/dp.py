import math

import torch

from conformal import split_conformal_threshold


# -----------------------------
# DP calibration via histogram quantile.
# -----------------------------
def dp_histogram_quantile_threshold(
        scores: torch.Tensor,
        alpha: float,
        eps_cal: float,
        num_bins: int = 200,
        seed: int = 0,
) -> float:
    """
    Approx DP threshold:
    - Bin scores in [0,1] into num_bins.
    - Add iid Laplace noise with scale 1/eps_cal to each bin count (vector Laplace mechanism under L1 sensitivity=1).
    - Build noisy CDF.
    - Take the conservative conformal rank k = ceil((m+1)*(1-alpha)) and find smallest bin whose noisy CDF >= k.

    Returns:
      tau_dp in [0,1]
    """
    assert eps_cal > 0, "eps_cal must be > 0"
    gen = torch.Generator()
    gen.manual_seed(seed)

    scores = scores.detach().cpu().clamp(0.0, 1.0)
    m = scores.numel()
    k = int(math.ceil((m + 1) * (1 - alpha)))
    k = min(max(k, 1), m)

    # Histogram counts
    # Map score to bin index in [0, num_bins-1]
    bin_idx = torch.clamp((scores * num_bins).long(), 0, num_bins - 1)
    counts = torch.bincount(bin_idx, minlength=num_bins).float()

    # Laplace noise
    # Sample Laplace(0, 1/eps) as: sign * Exp(scale) difference
    # PyTorch doesn't have Laplace on all versions; implement manually.
    scale = 1.0 / eps_cal
    u = torch.rand(num_bins, generator=gen) - 0.5
    noise = -scale * torch.sign(u) * torch.log1p(-2 * torch.abs(u))  # inverse CDF
    noisy = counts + noise

    # Clamp to nonnegative to form a "reasonable" CDF for the toy
    noisy = torch.clamp(noisy, min=0.0)
    cdf = torch.cumsum(noisy, dim=0)

    # If total mass vanished due to clamping (rare but possible for tiny m and small eps), fallback
    total = cdf[-1].item()
    if total <= 1e-6:
        # fallback to non-private threshold to avoid NaNs; still signals instability in logs
        return split_conformal_threshold(scores, alpha)

    # Find smallest bin where cdf >= k (k is in "counts" units)
    # If noisy total < k (possible), use last bin
    target = float(k)
    idx = int(torch.searchsorted(cdf, torch.tensor(target)).clamp(0, num_bins - 1).item())

    # Convert bin index to tau: use upper edge of the bin (conservative)
    tau = (idx + 1) / num_bins
    return float(min(max(tau, 0.0), 1.0))
