import math
from typing import Any, Dict, Tuple, Union

import torch

# -----------------------------
# DP calibration via histogram quantile.
# -----------------------------

def dp_histogram_quantile_threshold(
    scores: torch.Tensor,
    alpha: float,
    eps_cal: float,
    num_bins: int = 100,
    beta: float = 1e-3,
    seed: int = 0,
    return_info: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Differentially private conformal threshold via noisy cumulative counts
    on a fixed public grid.

    Idea:
    - Scores are assumed to lie in [0, 1].
    - Define a public grid t_b = b / B for b = 1, ..., B.
    - For each grid point, compute the cumulative count
          N_b = #{i : score_i <= t_b}.
    - Release a noisy version of the cumulative-count vector using iid
      Laplace noise with scale B / eps_cal, which gives eps_cal-DP under
      add/remove adjacency by the vector Laplace mechanism.
    - Select the first grid point whose noisy cumulative count exceeds the
      conservative target k + lambda.

    Formal guarantee:
    Let k = ceil((m + 1) * (1 - alpha)) and
        lambda = (B / eps_cal) * log(B / beta).
    Then, with probability at least 1 - beta, the selected threshold has
    true cumulative count at least k.
    For APS with the score used in conformal.py, one has the one-sided implication
        score(x, y) <= tau  ==>  y in APS_set_tau(x).

    Combined with exchangeability, this yields the formal lower bound
        P(Y in C_APS(X; tau_hat)) >= 1 - alpha - beta.

    Notes:
    - This mechanism is (intentionally, because I can't do better now) conservative.
    - Larger num_bins or smaller eps_cal increase the safety margin and
      therefore can increase prediction set size.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if eps_cal <= 0.0:
        raise ValueError(f"eps_cal must be > 0, got {eps_cal}")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")

    scores = scores.detach().cpu().float().clamp(0.0, 1.0)
    m = int(scores.numel())
    if m == 0:
        raise ValueError("scores must be non-empty")

    B = int(num_bins)
    k = int(math.ceil((m + 1) * (1.0 - alpha)))
    k = min(max(k, 1), m)

    # Public grid: t_b = b / B, b = 1, ..., B.
    grid = torch.arange(1, B + 1, dtype=torch.float32) / float(B)

    # Exact cumulative counts N_b = #{i : score_i <= t_b}.
    counts_cdf = (scores[:, None] <= grid[None, :]).sum(dim=0).float()

    # Vector Laplace mechanism with l1 sensitivity B.
    noise_scale = float(B) / float(eps_cal)
    gen = torch.Generator()
    gen.manual_seed(seed)
    noise = _laplace_noise((B,), scale=noise_scale, generator=gen)
    noisy_cdf = counts_cdf + noise

    # One-sided margin from a union bound over all B bins.
    lam = noise_scale * math.log(float(B) / float(beta))
    target = float(k) + float(lam)

    # Select the first grid point whose noisy cumulative count crosses the target.
    crossing = torch.nonzero(noisy_cdf >= target, as_tuple=False)
    if crossing.numel() == 0:
        idx = B - 1  # Fallback to tau = 1.
    else:
        idx = int(crossing[0].item())

    tau = float(grid[idx].item())

    if not return_info:
        return tau

    info: Dict[str, Any] = {
        "m": m,
        "k": k,
        "alpha": float(alpha),
        "beta": float(beta),
        "eps_cal": float(eps_cal),
        "num_bins": B,
        "noise_scale": float(noise_scale),
        "lambda": float(lam),
        "tau": float(tau),
    }
    return tau, info


def _laplace_noise(shape, scale: float, generator: torch.Generator) -> torch.Tensor:
    """
    Sample iid Laplace(0, scale) noise using inverse CDF sampling.
    """
    u = torch.rand(shape, generator=generator) - 0.5
    return -scale * torch.sign(u) * torch.log1p(-2 * torch.abs(u))


