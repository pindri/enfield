"""
Toy end-to-end demo: "contract -> (DP training) -> split conformal -> DP calibration threshold -> report"

What it does (fast sanity check, not a full paper implementation):
1) Train a NON-private MNIST classifier (tiny MLP), compute split-conformal threshold, evaluate coverage.
2) Train a DP classifier with Opacus (target epsilon), compute split-conformal threshold (non-private), evaluate coverage.
3) Using the DP-trained model, compute a DP calibration threshold via a DP histogram quantile (Laplace-noised histogram),
   then evaluate coverage and average prediction set size.
4) Write a report.json with key metrics.

Install (typical):
  pip install torch torchvision opacus

Run:
  python toy.py --device cuda
  python toy.py --dp_train_eps 4 --dp_cal_eps 1.0

Notes:
- This is a *toy feasibility test* meant to tell you in a few days if the idea "behaves sanely".
- DP calibration here uses a vector Laplace mechanism on histogram counts (iid Laplace(1/eps) per bin).
- Coverage "lower bound" is NOT fully rigorous in this toy; the goal is end-to-end behavior + artifact emission.
"""

import argparse
import json
import time
from dataclasses import asdict
from typing import Dict

from opacus import PrivacyEngine
from torch.utils.data import random_split
from torchvision import datasets, transforms

from conformal import aps_scores, split_conformal_threshold
from dp import dp_histogram_quantile_threshold
from utils import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs_np", type=int, default=3)
    ap.add_argument("--epochs_dp", type=int, default=3)

    # "Contract-like" knobs
    ap.add_argument("--coverage", type=float, default=0.90)
    ap.add_argument("--dp_train_eps", type=float, default=4.0)
    ap.add_argument("--dp_delta", type=float, default=1e-5)
    ap.add_argument("--dp_cal_eps", type=float, default=1.0)
    ap.add_argument("--num_bins", type=int, default=200)

    # Splits
    ap.add_argument("--cal_size", type=int, default=10000)  # from MNIST train(60k)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    set_seed(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[info] device={device}")

    # Data
    tfm = transforms.Compose([transforms.ToTensor()])
    ds_train_full = datasets.MNIST(root=args.outdir, train=True, download=True, transform=tfm)
    ds_test = datasets.MNIST(root=args.outdir, train=False, download=True, transform=tfm)

    # Split train_full into train + cal
    n_full = len(ds_train_full)  # 60000
    cal_size = min(args.cal_size, n_full // 2)  # keep it sane
    train_size = n_full - cal_size
    ds_train, ds_cal = random_split(
        ds_train_full,
        [train_size, cal_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device == "cuda"))
    cal_loader = DataLoader(ds_cal, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

    alpha = 1.0 - args.coverage
    contract = Contract(
        epsilon_train=float(args.dp_train_eps),
        delta=float(args.dp_delta),
        epsilon_cal=float(args.dp_cal_eps),
        coverage_target=float(args.coverage),
        alpha=float(alpha),
        num_bins=int(args.num_bins),
    )

    report: Dict = {
        "contract": asdict(contract),
        "meta": {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed,
            "device": device,
            "batch_size": args.batch_size,
            "epochs_np": args.epochs_np,
            "epochs_dp": args.epochs_dp,
            "cal_size": cal_size,
            "train_size": train_size,
        },
    }

    # -----------------------------
    # 1) Non-private baseline
    # -----------------------------
    print("\n[stage] non-private training")
    model_np = TinyMLP().to(device)
    opt_np = torch.optim.SGD(model_np.parameters(), lr=0.2, momentum=0.9)

    for ep in range(args.epochs_np):
        loss = train_epoch(model_np, train_loader, opt_np, device)
        acc = accuracy(model_np, test_loader, device)
        print(f"[np] epoch={ep+1}/{args.epochs_np} loss={loss:.4f} test_acc={acc:.4f}")

    # Non-private conformal threshold
    probs_cal_np, y_cal = predict_proba(model_np, cal_loader, device)
    # scores_cal_np = conformity_scores(probs_cal_np, y_cal)
    # tau_np = split_conformal_threshold(scores_cal_np, alpha)
    # eval_np = evaluate_coverage(model_np, test_loader, tau_np, device)

    scores_cal_np = aps_scores(probs_cal_np, y_cal)
    print("score_stats:",
          "min", float(scores_cal_np.min()),
          "median", float(scores_cal_np.median()),
          "p90", float(scores_cal_np.kthvalue(int(0.9*len(scores_cal_np))).values),
          "max", float(scores_cal_np.max()))
    tau_np = split_conformal_threshold(scores_cal_np, alpha)
    eval_np = evaluate_coverage_aps(model_np, test_loader, tau_np, device)

    print("min_set_size_check:",
          "min_prob_sum", float(probs_cal_np.sum(dim=1).min()),
          "max_prob_sum", float(probs_cal_np.sum(dim=1).max()),
          "min_maxprob", float(probs_cal_np.max(dim=1).values.min()))

    report["non_private"] = {
        "tau": tau_np,
        "test_accuracy": accuracy(model_np, test_loader, device),
        "test_coverage": eval_np["coverage"],
        "avg_set_size": eval_np["avg_set_size"],
    }
    print(f"[np] tau={tau_np:.4f} coverage={eval_np['coverage']:.4f} avg_set_size={eval_np['avg_set_size']:.3f}")

    # -----------------------------
    # 2) DP training (Opacus)
    # -----------------------------
    print("\n[stage] dp training (opacus)")
    model_dp = TinyMLP().to(device)
    opt_dp = torch.optim.SGD(model_dp.parameters(), lr=0.2, momentum=0.9)

    privacy_engine = PrivacyEngine()

    # This helper chooses a noise multiplier to hit target epsilon for given epochs.
    # It uses sample_rate from loader. Great for toy testing.
    model_dp, opt_dp, train_loader_dp = privacy_engine.make_private_with_epsilon(
        module=model_dp,
        optimizer=opt_dp,
        data_loader=train_loader,
        epochs=args.epochs_dp,
        target_epsilon=args.dp_train_eps,
        target_delta=args.dp_delta,
        max_grad_norm=1.0,
    )

    for ep in range(args.epochs_dp):
        loss = train_epoch(model_dp, train_loader_dp, opt_dp, device, dp=True)
        # eps_spent = privacy_engine.get_epsilon(args.dp_delta)
        eps_spent = privacy_engine.accountant.get_epsilon(delta=args.dp_delta)
        acc = accuracy(model_dp, test_loader, device)
        print(f"[dp] epoch={ep+1}/{args.epochs_dp} loss={loss:.4f} eps={eps_spent:.3f} test_acc={acc:.4f}")

    # eps_realized = float(privacy_engine.get_epsilon(args.dp_delta))
    eps_realized = float(
        privacy_engine.accountant.get_epsilon(delta=args.dp_delta)
    )
    noise_multiplier = float(opt_dp.noise_multiplier)

    # Conformal on DP model (non-private threshold first)
    probs_cal_dp, y_cal2 = predict_proba(model_dp, cal_loader, device)
    # scores_cal_dp = conformity_scores(probs_cal_dp, y_cal2)
    # tau_dp_nonpriv_cal = split_conformal_threshold(scores_cal_dp, alpha)
    # eval_dp_nonpriv_cal = evaluate_coverage(model_dp, test_loader, tau_dp_nonpriv_cal, device)
    scores_cal_dp = aps_scores(probs_cal_dp, y_cal2)
    tau_dp_nonpriv_cal = split_conformal_threshold(scores_cal_dp, alpha)
    eval_dp_nonpriv_cal = evaluate_coverage_aps(model_dp, test_loader, tau_dp_nonpriv_cal, device)

    report["dp_training"] = {
        "epsilon_target": float(args.dp_train_eps),
        "epsilon_realized": eps_realized,
        "delta": float(args.dp_delta),
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": 1.0,
        "tau_nonprivate_cal": tau_dp_nonpriv_cal,
        "test_accuracy": accuracy(model_dp, test_loader, device),
        "test_coverage_nonprivate_cal": eval_dp_nonpriv_cal["coverage"],
        "avg_set_size_nonprivate_cal": eval_dp_nonpriv_cal["avg_set_size"],
    }
    print(f"[dp] eps_realized={eps_realized:.3f} noise_multiplier={noise_multiplier:.3f}")
    print(f"[dp+nonprivcal] tau={tau_dp_nonpriv_cal:.4f} coverage={eval_dp_nonpriv_cal['coverage']:.4f} avg_set_size={eval_dp_nonpriv_cal['avg_set_size']:.3f}")

    # -----------------------------
    # 3) DP calibration threshold via DP histogram quantile
    # -----------------------------
    print("\n[stage] dp calibration (dp histogram quantile threshold)")
    tau_dp_cal = dp_histogram_quantile_threshold(
        scores=scores_cal_dp,
        alpha=alpha,
        eps_cal=args.dp_cal_eps,
        num_bins=args.num_bins,
        seed=args.seed,
    )
    # eval_dp_dpcal = evaluate_coverage(model_dp, test_loader, tau_dp_cal, device)
    eval_dp_dpcal = evaluate_coverage_aps(model_dp, test_loader, tau_dp_cal, device)

    # Not a fully rigorous bound here—just a placeholder to keep artifact structure.
    # For the feasibility test you care about "does it behave sanely?".
    coverage_lb_placeholder = max(0.0, eval_dp_dpcal["coverage"] - 0.02)

    report["dp_calibration"] = {
        "epsilon_cal": float(args.dp_cal_eps),
        "num_bins": int(args.num_bins),
        "tau_dp_cal": tau_dp_cal,
        "test_coverage_dp_cal": eval_dp_dpcal["coverage"],
        "avg_set_size_dp_cal": eval_dp_dpcal["avg_set_size"],
        "coverage_lower_bound_placeholder": coverage_lb_placeholder,
        "note": "coverage_lower_bound_placeholder is NOT a formal DP-adjusted bound; replace with your derived bound later.",
    }

    print(f"[dpcal] tau_dp={tau_dp_cal:.4f} coverage={eval_dp_dpcal['coverage']:.4f} avg_set_size={eval_dp_dpcal['avg_set_size']:.3f}")
    if eval_dp_dpcal["avg_set_size"] > 7.0:
        print("[warn] avg_set_size is very large; DP calibration may be too noisy or model too weak. Try larger dp_cal_eps or larger cal_size.")

    # -----------------------------
    # 4) Contract-style PASS/FAIL checks (toy)
    # -----------------------------
    report["pass_fail"] = {
        "privacy_ok": eps_realized <= args.dp_train_eps + 1e-6,
        "coverage_ok_empirical": eval_dp_dpcal["coverage"] >= args.coverage - 0.02,  # loose tolerance for toy
        "overall_toy_ok": (eps_realized <= args.dp_train_eps + 1e-6) and (eval_dp_dpcal["coverage"] >= args.coverage - 0.02),
    }

    # Lightweight "hashes" (toy)
    contract_text = json.dumps(report["contract"], sort_keys=True)
    report["hashes"] = {
        "contract_sha256": sha256_of_text(contract_text),
        "script_sha256": sha256_of_text(open(__file__, "r", encoding="utf-8").read()),
    }

    outpath = os.path.join(args.outdir, "report.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"\n[done] wrote {outpath}")
    print("[next] for a quick sanity sweep, try:")
    print("  python toy_dp_conformal_mnist.py --dp_train_eps 8 --dp_cal_eps 2.0")
    print("  python toy_dp_conformal_mnist.py --dp_train_eps 4 --dp_cal_eps 1.0")
    print("  python toy_dp_conformal_mnist.py --dp_train_eps 2 --dp_cal_eps 0.5")


if __name__ == "__main__":
    main()
