"""
Initial, toy-ish implementation.

DP training + APS split conformal + formal DP calibration via
Laplace-noisy cumulative counts on a fixed score grid.

Formal calibration guarantee:
    P(Y in C_APS(X; tau_hat)) >= 1 - alpha - beta
under exchangeability.

Implemented idea:
1) Train a NON-private MNIST classifier (with tiny MLP), compute a split-conformal threshold, evaluate coverage.
2) Train a DP classifier (for a target eps), compute a split-conformal threshold (non-private), evaluate coverage.
3) Using the DP-trained model, compute a DP calibration threshold via a DP histogram quantile
   (using Laplace-noised histogram), then evaluate coverage and average prediction set size.
4) Write a report.json with key metrics.
"""

import argparse
import json
import math
import time
from dataclasses import asdict

from opacus import PrivacyEngine
from torch.utils.data import random_split
from torchvision import datasets, transforms

from conformal import aps_scores, split_conformal_threshold, summarize_scores, save_score_hist
from dp import dp_histogram_quantile_threshold
from utils import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="toy_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs_np", type=int, default=3)
    ap.add_argument("--epochs_dp", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dataset", type=str, default="cifar10",
                choices=["mnist", "fashionmnist", "cifar10"])

    # "Contract-like" made up knobs.
    # ap.add_argument("--coverage", type=float, default=0.90)
    ap.add_argument("--coverage_target", type=float, default=0.90)
    ap.add_argument("--nominal_coverage", type=float, default=0.91)
    ap.add_argument("--dp_train_eps", type=float, default=4.0)
    ap.add_argument("--dp_delta", type=float, default=1e-5)
    ap.add_argument("--dp_cal_eps", type=float, default=1.0)
    ap.add_argument("--num_bins", type=int, default=50)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=1.0)


    # Calibration split.
    ap.add_argument("--cal_size", type=int, default=10000)  # MNIST has 60K data points.

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.data_dir)
    make_reproducible(args.seed)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    # Data.
    if args.dataset in ["mnist", "fashionmnist"]:
        tfm = transforms.Compose([transforms.ToTensor()])
        DatasetClass = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    else:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        DatasetClass = datasets.CIFAR10

    ds_train_full = DatasetClass(root=args.data_dir, train=True, download=True, transform=tfm)
    ds_test = DatasetClass(root=args.data_dir, train=False, download=True, transform=tfm)

    n_full = len(ds_train_full)
    cal_size = min(args.cal_size, n_full // 2)
    train_size = n_full - cal_size
    ds_train, ds_cal = random_split(
        ds_train_full,
        [train_size, cal_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # TODO: Not quite so sure how to best pick num_workers.
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device == "cuda"))
    cal_loader = DataLoader(ds_cal, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=(device == "cuda"))
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=(device == "cuda"))

    nominal_coverage = float(args.nominal_coverage)
    requested_coverage_target = float(args.coverage_target)

    alpha = 1.0 - nominal_coverage
    coverage_lb_formal = max(0.0, 1.0 - alpha - args.beta)

    contract = Contract(
        epsilon_train=float(args.dp_train_eps),
        delta=float(args.dp_delta),
        epsilon_cal=float(args.dp_cal_eps),
        # coverage_target=float(coverage_lb_formal),
        coverage_target=requested_coverage_target,
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
            "nominal_coverage": nominal_coverage,
            "coverage_target": requested_coverage_target,
            "beta": float(args.beta),
        },
    }


    if args.dataset == "cifar10":
        model_np = TinyCNN(in_channels=3, num_classes=10).to(device)
        model_dp = TinyCNN(in_channels=3, num_classes=10).to(device)
    else:
        model_np = TinyMLP().to(device)
        model_dp = TinyMLP().to(device)

    # -----------------------------
    # Non-private baseline.
    # -----------------------------
    print("\n[stage] non-private training")
    # model_np = TinyMLP().to(device)
    if args.dataset == "cifar10":
        opt_np = torch.optim.Adam(model_np.parameters(), lr=1e-3)
    else:
        opt_np = torch.optim.SGD(model_np.parameters(), lr=0.2, momentum=0.9)


    for ep in range(args.epochs_np):
        loss = train_epoch(model_np, train_loader, opt_np, device)
        acc = accuracy(model_np, test_loader, device)
        print(f"[np] epoch={ep+1}/{args.epochs_np} loss={loss:.4f} test_acc={acc:.4f}")

    # Non-private conformal threshold.
    probs_cal_np, y_cal = predict_proba(model_np, cal_loader, device, temperature=args.temperature)
    scores_cal_np = aps_scores(probs_cal_np, y_cal)
    summarize_scores("np_scores", scores_cal_np)
    print("score_stats:",
          "min", float(scores_cal_np.min()),
          "median", float(scores_cal_np.median()),
          "p90", float(scores_cal_np.kthvalue(int(0.9*len(scores_cal_np))).values),
          "max", float(scores_cal_np.max()))
    tau_np = split_conformal_threshold(scores_cal_np, alpha)
    # eval_np = evaluate_coverage_aps(model_np, test_loader, tau_np, device)
    eval_np = evaluate_coverage_aps(model_np, test_loader, tau_np, device, temperature=args.temperature)

    if args.verbose:
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
    # DP training.
    # -----------------------------
    print("\n[stage] dp training")
    # model_dp = TinyMLP().to(device)
    if args.dataset == "cifar10":
        opt_dp = torch.optim.SGD(model_dp.parameters(), lr=0.02, momentum=0.9)
    else:
        opt_dp = torch.optim.SGD(model_dp.parameters(), lr=0.2, momentum=0.9)

    privacy_engine = PrivacyEngine()

    # Helper to get a noise multiplier to hit target epsilon for given epochs.
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
        loss = train_epoch(model_dp, train_loader_dp, opt_dp, device)
        eps_spent = privacy_engine.accountant.get_epsilon(delta=args.dp_delta)
        acc = accuracy(model_dp, test_loader, device)
        print(f"[dp] epoch={ep+1}/{args.epochs_dp} loss={loss:.4f} eps={eps_spent:.3f} test_acc={acc:.4f}")

    eps_realized = float(privacy_engine.accountant.get_epsilon(delta=args.dp_delta))
    noise_multiplier = float(opt_dp.noise_multiplier)

    # Conformal on the DP model (non-private threshold first).
    probs_cal_dp, y_cal2 = predict_proba(model_dp, cal_loader, device, temperature=args.temperature)
    scores_cal_dp = aps_scores(probs_cal_dp, y_cal2)
    summarize_scores("dp_scores", scores_cal_dp)
    m = scores_cal_dp.numel()
    k = int(math.ceil((m + 1) * (1 - alpha)))
    k = min(max(k, 1), m)
    sorted_scores, _ = torch.sort(scores_cal_dp)
    print(
        f"[dp_nonpriv_quantile] k={k}/{m} "
        f"score_k={sorted_scores[k-1].item():.6f} "
        f"score_(k-5)={sorted_scores[max(k-6,0)].item():.6f} "
        f"score_(k+5)={sorted_scores[min(k+4,m-1)].item():.6f}"
    )
    tau_dp_nonpriv_cal = split_conformal_threshold(scores_cal_dp, alpha)
    # eval_dp_nonpriv_cal = evaluate_coverage_aps(model_dp, test_loader, tau_dp_nonpriv_cal, device)
    eval_dp_nonpriv_cal = evaluate_coverage_aps(model_dp, test_loader, tau_dp_nonpriv_cal, device, temperature=args.temperature)

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
    # 3) DP calibration threshold via DP histogram quantile.
    # -----------------------------
    print("\n[stage] dp calibration (formal noisy cumulative-count quantile)")
    tau_dp_cal, dpcal_info = dp_histogram_quantile_threshold(
        scores=scores_cal_dp,
        alpha=alpha,
        eps_cal=args.dp_cal_eps,
        num_bins=args.num_bins,
        beta=args.beta,
        seed=args.seed,
        return_info=True,
    )
    # eval_dp_dpcal = evaluate_coverage_aps(model_dp, test_loader, tau_dp_cal, device)
    eval_dp_dpcal = evaluate_coverage_aps(model_dp, test_loader, tau_dp_cal, device, temperature=args.temperature)

    report["dp_calibration"] = {
        "epsilon_cal": float(args.dp_cal_eps),
        "nominal_coverage": nominal_coverage,
        "coverage_target": requested_coverage_target,
        "beta": float(args.beta),
        "num_bins": int(args.num_bins),
        "k": int(dpcal_info["k"]),
        "lambda": float(dpcal_info["lambda"]),
        "noise_scale": float(dpcal_info["noise_scale"]),
        "tau_dp_cal": float(tau_dp_cal),
        "test_coverage_dp_cal": eval_dp_dpcal["coverage"],
        "avg_set_size_dp_cal": eval_dp_dpcal["avg_set_size"],
        "coverage_lower_bound_formal": coverage_lb_formal,
        "requested_coverage_target": requested_coverage_target,
        "mechanism": "Laplace-noisy cumulative counts on a fixed public grid",
        "guarantee": "P(Y in C_APS(X; tau_hat)) >= 1 - alpha - beta",
    }

    print(f"[dpcal] tau_dp={tau_dp_cal:.4f} coverage={eval_dp_dpcal['coverage']:.4f} avg_set_size={eval_dp_dpcal['avg_set_size']:.3f}")
    if eval_dp_dpcal["avg_set_size"] > 7.0:
        print("[warn] avg_set_size is very large; DP calibration may be too noisy or model too weak. Try larger dp_cal_eps or larger cal_size.")

    report["privacy_composition"] = {
        "epsilon_train_realized": eps_realized,
        "epsilon_cal": float(args.dp_cal_eps),
        "epsilon_total_basic_composition": eps_realized + float(args.dp_cal_eps),
        "delta_total_basic_composition": float(args.dp_delta),
    }

    # -----------------------------
    # 4) Contract-style PASS/FAIL checks.
    # -----------------------------
    coverage_bound_formal_ok = coverage_lb_formal >= requested_coverage_target
    coverage_ok_empirical = eval_dp_dpcal["coverage"] >= requested_coverage_target

    report["pass_fail"] = {
        "privacy_training_ok": eps_realized <= args.dp_train_eps + 1e-6,
        "privacy_total_basic_composition_ok":
            (eps_realized + float(args.dp_cal_eps)) <= (args.dp_train_eps + float(args.dp_cal_eps) + 1e-6),
        # Formal check: theorem-certified lower bound meets the requested target.
        "coverage_bound_formal_ok": coverage_bound_formal_ok,
        # Empirical check: observed test coverage meets the requested target.
        "coverage_ok_empirical": coverage_ok_empirical,
        # Formal feasibility should use the formal bound, not empirical coverage.
        "overall_formal_ok":
            (eps_realized <= args.dp_train_eps + 1e-6)
            and coverage_bound_formal_ok,
        # Optional extra diagnostic.
        "overall_empirical_ok":
            (eps_realized <= args.dp_train_eps + 1e-6)
            and coverage_ok_empirical,
    }

    # Lightweight, made up "hashes" for things to look proper.
    contract_text = json.dumps(report["contract"], sort_keys=True)
    report["hashes"] = {
        "contract_sha256": sha256_of_text(contract_text),
        "script_sha256": sha256_of_text(open(__file__, "r", encoding="utf-8").read()),
    }

    # Some histogramming because things are weird.
    save_score_hist(scores_cal_np, f"{args.out_dir}/np_scores_seed_{args.seed}.png", "NP APS scores")
    save_score_hist(scores_cal_dp, f"{args.out_dir}/dp_scores_seed_{args.seed}.png", "DP APS scores")

    outpath = os.path.join(
        args.out_dir,
        f"report_traineps_{args.dp_train_eps}"
        f"_caleps_{args.dp_cal_eps}"
        f"_calsize_{args.cal_size}"
        f"_nomcov_{args.nominal_coverage}"
        f"_target_{args.coverage_target}"
        f"_beta_{args.beta}"
        f"_seed_{args.seed}.json"
    )
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"\n[done] wrote {outpath}")

    # Some APS debugging.
    sorted_probs, sorted_idx = torch.sort(probs_cal_dp, dim=1, descending=True)
    matches = (sorted_idx == y_cal2[:, None])
    pos = matches.float().argmax(dim=1).float()

    print(
        "[true_label_rank]",
        "median", pos.median().item(),
        "p90", torch.quantile(pos, 0.9).item(),
        "max", pos.max().item(),
    )

    true_probs = probs_cal_dp[torch.arange(probs_cal_dp.size(0)), y_cal2]
    top1_probs = probs_cal_dp.max(dim=1).values

    print(
        "[prob_stats]",
        "true_prob_median", true_probs.median().item(),
        "true_prob_p10", torch.quantile(true_probs, 0.1).item(),
        "top1_prob_median", top1_probs.median().item(),
        "top1_prob_p90", torch.quantile(top1_probs, 0.9).item(),
    )


if __name__ == "__main__":
    main()
