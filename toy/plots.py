import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from utils import ensure_dir, make_reproducible


def filter_df(
    df: pd.DataFrame,
    nominal_coverage: float | None = None,
    coverage_target: float | None = None,
    beta: float | None = None,
    cal_size: int | None = None,
) -> pd.DataFrame:
    dff = df.copy()
    if nominal_coverage is not None:
        dff = dff[np.isclose(dff["nominal_coverage"], nominal_coverage)]
    if coverage_target is not None:
        dff = dff[np.isclose(dff["coverage_target"], coverage_target)]
    if beta is not None:
        dff = dff[np.isclose(dff["beta"], beta)]
    if cal_size is not None:
        dff = dff[dff["cal_size"] == cal_size]
    return dff


def load_reports(report_dir: str | Path) -> pd.DataFrame:
    """
    Load all report_*.json files and flatten the main fields used for plotting.
    """
    report_dir = Path(report_dir)
    rows = []

    for path in sorted(report_dir.glob("report_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)

        row = {
            "file": path.name,

            # Contract.
            "alpha": r["contract"]["alpha"],
            "coverage_target": r["contract"]["coverage_target"],
            "delta": r["contract"]["delta"],
            "epsilon_train_target": r["contract"]["epsilon_train"],
            "epsilon_cal": r["contract"]["epsilon_cal"],
            "num_bins": r["contract"]["num_bins"],

            # Meta.
            "cal_size": r["meta"]["cal_size"],
            "train_size": r["meta"]["train_size"],
            "seed": r["meta"]["seed"],
            "device": r["meta"]["device"],
            "nominal_coverage": r["meta"]["nominal_coverage"],

            # Non-private baseline.
            "np_accuracy": r["non_private"]["test_accuracy"],
            "np_coverage": r["non_private"]["test_coverage"],
            "np_avg_set_size": r["non_private"]["avg_set_size"],
            "np_tau": r["non_private"]["tau"],

            # DP training.
            "dp_accuracy": r["dp_training"]["test_accuracy"],
            "epsilon_train_realized": r["dp_training"]["epsilon_realized"],
            "noise_multiplier": r["dp_training"]["noise_multiplier"],
            "tau_nonprivate_cal": r["dp_training"]["tau_nonprivate_cal"],
            "test_coverage_nonprivate_cal": r["dp_training"]["test_coverage_nonprivate_cal"],
            "dp_avg_set_size_nonprivate_cal": r["dp_training"]["avg_set_size_nonprivate_cal"],

            # DP composition.
            "epsilon_total_basic_composition": r["privacy_composition"]["epsilon_total_basic_composition"],

            # DP calibration.
            "beta": r["dp_calibration"]["beta"],
            "coverage_lower_bound_formal": r["dp_calibration"]["coverage_lower_bound_formal"],
            "tau_dp_cal": r["dp_calibration"]["tau_dp_cal"],
            "test_coverage_dp_cal": r["dp_calibration"]["test_coverage_dp_cal"],
            "avg_set_size_dp_cal": r["dp_calibration"]["avg_set_size_dp_cal"],
            "lambda": r["dp_calibration"]["lambda"],
            "noise_scale": r["dp_calibration"]["noise_scale"],
            "k": r["dp_calibration"]["k"],

            # Pass/Fail.
            "coverage_bound_formal_ok": r["pass_fail"]["coverage_bound_formal_ok"],
            "privacy_training_ok": r["pass_fail"]["privacy_training_ok"],
            "coverage_empirical_ok": r["pass_fail"]["coverage_empirical_ok"],
            "overall_formal_ok": r["pass_fail"]["overall_formal_ok"],
            "overall_empirical_ok": r["pass_fail"]["overall_empirical_ok"],
        }

        # Useful derived fields.
        row["empirical_margin_to_bound"] = (
            row["test_coverage_dp_cal"] - row["coverage_lower_bound_formal"]
        )
        row["empirical_margin_to_target"] = (
            row["test_coverage_dp_cal"] - row["coverage_target"]
        )

        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No report_*.json files found in {report_dir}")

    df = pd.DataFrame(rows)
    return df


def save_dataframe(df: pd.DataFrame, out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def plot_coverage_vs_formal_bound(df: pd.DataFrame, outpath: str | Path) -> None:
    """
    Scatter: x = formal lower bound, y = empirical coverage under DP calibration.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(
        df["coverage_lower_bound_formal"],
        df["test_coverage_dp_cal"],
    )

    lo = min(df["coverage_lower_bound_formal"].min(), df["test_coverage_dp_cal"].min())
    hi = max(df["coverage_lower_bound_formal"].max(), df["test_coverage_dp_cal"].max())
    margin = 0.01
    lo = max(0.0, lo - margin)
    hi = min(1.0, hi + margin)

    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Formal coverage lower bound")
    plt.ylabel("Empirical coverage (DP calibration)")
    plt.title("Empirical coverage vs formal bound")
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_contract_heatmap(
    df: pd.DataFrame,
    outpath: str | Path,
    value_col: str,
    beta: float,
    cal_size: int | None = None,
    epsilon_train_target: float | None = None,
    epsilon_cal: float | None = None,
    aggregate: str = "mean",
) -> None:
    dff = df.copy()
    dff = dff[np.isclose(dff["beta"], beta)]
    if cal_size is not None:
        dff = dff[dff["cal_size"] == cal_size]

    if epsilon_train_target is not None:
        dff = dff[np.isclose(dff["epsilon_train_target"], epsilon_train_target)]
    if epsilon_cal is not None:
        dff = dff[np.isclose(dff["epsilon_cal"], epsilon_cal)]

    if dff.empty:
        raise ValueError("No rows remain after filtering")

    grouped = dff.groupby(
        ["nominal_coverage", "coverage_target"], as_index=False
    )[value_col]

    if aggregate == "mean":
        agg = grouped.mean()
    elif aggregate == "median":
        agg = grouped.median()
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")

    pivot = agg.pivot(
        index="nominal_coverage",
        columns="coverage_target",
        values=value_col,
    ).sort_index().sort_index(axis=1)

    plt.figure(figsize=(7, 5))
    vmin = min(0, np.nanmin(pivot.values))
    vmax = min(1, np.nanmax(pivot.values))
    norm = TwoSlopeNorm(vmin=vmin - 1e-10, vcenter=0.0, vmax=vmax + 1e-10)
    im = plt.imshow(pivot.values, aspect="auto", interpolation="nearest",
                    cmap="RdYlGn", norm=norm,
                    # vmin=-1.0, vmax=1.0, cmap="viridis"
                    )

    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(y) for y in pivot.index])

    plt.xlabel("Coverage target")
    plt.ylabel("Nominal coverage")
    plt.title(value_col)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            txt = "NA" if pd.isna(val) else f"{float(val):.3f}"
            plt.text(j, i, txt, ha="center", va="center")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_coverage_vs_epsilon_cal(
    df: pd.DataFrame,
    outpath: str | Path,
    fixed_cal_size: int | None = None,
) -> None:
    """
    Line plot of avg coverage vs epsilon_cal, with one curve per train epsilon target.
    """
    dff = df.copy()
    if fixed_cal_size is not None:
        dff = dff[dff["cal_size"] == fixed_cal_size].copy()

    if dff.empty:
        raise ValueError("No rows remain after filtering for plot_set_size_vs_epsilon_cal")

    plt.figure(figsize=(7, 5))

    for eps_train in sorted(dff["epsilon_train_target"].unique()):
        g = dff[dff["epsilon_train_target"] == eps_train].sort_values("epsilon_cal")
        plt.plot(
            g["epsilon_cal"],
            g["test_coverage_dp_cal"],
            marker="o",
            label=f"train eps={eps_train}",
        )

    plt.xlabel(r"$\epsilon_{\mathrm{cal}}$")
    plt.ylabel("Empirical coverage")
    plt.title("Coverage vs calibration privacy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_set_size_vs_epsilon_cal_by_nominal(
    df: pd.DataFrame,
    outpath: str | Path,
    fixed_cal_size: int | None = None,
    fixed_epsilon_train_target: float | None = None,
    fixed_coverage_target: float | None = None,
    aggregate: str = "mean",
) -> None:
    """
    Line plot of avg set size vs epsilon_cal, with one curve per nominal coverage.
    If multiple rows remain per epsilon_cal (e.g. multiple seeds), aggregate them.
    """
    dff = df.copy()

    if fixed_cal_size is not None:
        dff = dff[dff["cal_size"] == fixed_cal_size].copy()

    if fixed_epsilon_train_target is not None:
        dff = dff[np.isclose(dff["epsilon_train_target"], fixed_epsilon_train_target)].copy()

    if fixed_coverage_target is not None:
        dff = dff[np.isclose(dff["coverage_target"], fixed_coverage_target)].copy()

    if dff.empty:
        raise ValueError("No rows remain after filtering for plot_set_size_vs_epsilon_cal_by_nominal")

    plt.figure(figsize=(7, 5))

    for nomcov in sorted(dff["nominal_coverage"].unique()):
        g = dff[np.isclose(dff["nominal_coverage"], nomcov)].copy()

        grouped = g.groupby("epsilon_cal", as_index=False)["avg_set_size_dp_cal"]
        if aggregate == "mean":
            gplot = grouped.mean()
        elif aggregate == "median":
            gplot = grouped.median()
        else:
            raise ValueError("aggregate must be 'mean' or 'median'")

        gplot = gplot.sort_values("epsilon_cal")

        plt.plot(
            gplot["epsilon_cal"],
            gplot["avg_set_size_dp_cal"],
            marker="o",
            label=f"nom={nomcov}",
        )

    plt.xlabel(r"$\epsilon_{\mathrm{cal}}$")
    plt.ylabel("Average prediction set size")
    plt.title("Set size vs calibration privacy")
    plt.legend(title="Nominal coverage")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_coverage_vs_epsilon_cal_by_nominal(
    df: pd.DataFrame,
    outpath: str | Path,
    fixed_cal_size: int | None = None,
    fixed_epsilon_train_target: float | None = None,
    fixed_coverage_target: float | None = None,
    aggregate: str = "mean",
) -> None:
    """
    Line plot of empirical coverage vs epsilon_cal, with one curve per nominal coverage.
    If multiple rows remain per epsilon_cal (e.g. multiple seeds), aggregate them.
    """
    dff = df.copy()

    if fixed_cal_size is not None:
        dff = dff[dff["cal_size"] == fixed_cal_size].copy()

    if fixed_epsilon_train_target is not None:
        dff = dff[np.isclose(dff["epsilon_train_target"], fixed_epsilon_train_target)].copy()

    if fixed_coverage_target is not None:
        dff = dff[np.isclose(dff["coverage_target"], fixed_coverage_target)].copy()

    if dff.empty:
        raise ValueError("No rows remain after filtering for plot_coverage_vs_epsilon_cal_by_nominal")

    plt.figure(figsize=(7, 5))

    for nomcov in sorted(dff["nominal_coverage"].unique()):
        g = dff[np.isclose(dff["nominal_coverage"], nomcov)].copy()

        grouped = g.groupby("epsilon_cal", as_index=False)["test_coverage_dp_cal"]
        if aggregate == "mean":
            gplot = grouped.mean()
        elif aggregate == "median":
            gplot = grouped.median()
        else:
            raise ValueError("aggregate must be 'mean' or 'median'")

        gplot = gplot.sort_values("epsilon_cal")

        plt.plot(
            gplot["epsilon_cal"],
            gplot["test_coverage_dp_cal"],
            marker="o",
            label=f"nom={nomcov}",
        )

    plt.xlabel(r"$\epsilon_{\mathrm{cal}}$")
    plt.ylabel("Empirical coverage")
    plt.title("Coverage vs calibration privacy")
    plt.legend(title="Nominal coverage")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_set_size_vs_epsilon_cal(
    df: pd.DataFrame,
    outpath: str | Path,
    fixed_cal_size: int | None = None,
) -> None:
    """
    Line plot of avg set size vs epsilon_cal, with one curve per train epsilon target.
    """
    dff = df.copy()
    if fixed_cal_size is not None:
        dff = dff[dff["cal_size"] == fixed_cal_size].copy()

    if dff.empty:
        raise ValueError("No rows remain after filtering for plot_set_size_vs_epsilon_cal")

    plt.figure(figsize=(7, 5))

    for eps_train in sorted(dff["epsilon_train_target"].unique()):
        g = dff[dff["epsilon_train_target"] == eps_train].sort_values("epsilon_cal")
        plt.plot(
            g["epsilon_cal"],
            g["avg_set_size_dp_cal"],
            marker="o",
            label=f"train eps={eps_train}",
        )

    plt.xlabel(r"$\epsilon_{\mathrm{cal}}$")
    plt.ylabel("Average prediction set size")
    plt.title("Set size vs calibration privacy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_accuracy_vs_epsilon_train(
    df: pd.DataFrame,
    outpath: str | Path,
    fixed_cal_size: int | None = None,
) -> None:
    """
    Line plot of DP model accuracy vs realized or target training epsilon.
    One curve per calibration epsilon.
    """
    dff = df.copy()
    if fixed_cal_size is not None:
        dff = dff[dff["cal_size"] == fixed_cal_size].copy()

    if dff.empty:
        raise ValueError("No rows remain after filtering for plot_accuracy_vs_epsilon_train")

    plt.figure(figsize=(7, 5))

    for eps_cal in sorted(dff["epsilon_cal"].unique()):
        g = dff[dff["epsilon_cal"] == eps_cal].sort_values("epsilon_train_realized")
        plt.plot(
            g["epsilon_train_realized"],
            g["dp_accuracy"],
            marker="o",
            label=f"cal eps={eps_cal}",
        )

    plt.xlabel(r"Realized $\epsilon_{\mathrm{train}}$")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs training privacy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_contract_feasibility_map(
    df: pd.DataFrame,
    outpath: str | Path,
    value_col: str,
    beta: float,
    cal_size: int | None = None,
    epsilon_train_target: float | None = None,
    epsilon_cal: float | None = None,
    aggregate: str = "all",
    mask_formally_infeasible: bool = False,
) -> None:

    dff = df.copy()
    dff = dff[np.isclose(dff["beta"], beta)]
    if cal_size is not None:
        dff = dff[dff["cal_size"] == cal_size]

    if epsilon_train_target is not None:
        dff = dff[np.isclose(dff["epsilon_train_target"], epsilon_train_target)]
    if epsilon_cal is not None:
        dff = dff[np.isclose(dff["epsilon_cal"], epsilon_cal)]

    if dff.empty:
        raise ValueError("No rows remain after filtering")

    # To exclude absolutely impossible settings from the plot.
    # if filter_impossible:
    #     dff = dff[dff["nominal_coverage"] - dff["beta"] >= dff["coverage_target"]]
    if mask_formally_infeasible:
        dff = dff[dff["coverage_lower_bound_formal"] >= dff["coverage_target"]]

    grouped = dff.groupby(
        ["nominal_coverage", "coverage_target"], as_index=False
    )[value_col]

    if aggregate == "all":
        agg = grouped.all()
    elif aggregate == "mean":
        agg = grouped.mean()
    else:
        raise ValueError("aggregate must be 'all' or 'mean'")

    pivot = agg.pivot(
        index="nominal_coverage",
        columns="coverage_target",
        values=value_col,
    ).sort_index().sort_index(axis=1)

    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest",
               vmin=-1.0, vmax=1.0, cmap="viridis"
               )

    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(y) for y in pivot.index])

    plt.xlabel("Coverage target")
    plt.ylabel("Nominal coverage")
    plt.title(value_col)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                txt = "NA"
            elif aggregate == "mean":
                txt = f"{float(val):.2f}"
            else:
                txt = "1" if bool(val) else "0"
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()



def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--analysis_out_dir", type=str, default="analysis")
    ap.add_argument("--out_dir", type=str, default="toy_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.data_dir)
    ensure_dir(args.analysis_out_dir)

    make_reproducible(args.seed)

    # Check and create folders.
    report_dir = args.out_dir
    analysis_out_dir = args.analysis_out_dir
    data_dir = args.data_dir

    df = load_reports(report_dir)
    save_dataframe(df, f"{analysis_out_dir}/all_reports.csv")

    if args.verbose:
        print("Loaded", len(df), "reports")
        print(df[
            [
                "file",
                "epsilon_train_target",
                "epsilon_train_realized",
                "epsilon_cal",
                "coverage_target",
                "nominal_coverage",
                "coverage_lower_bound_formal",
                "test_coverage_dp_cal",
                "avg_set_size_dp_cal",
                "dp_accuracy",
                "overall_formal_ok",
            ]
        ].sort_values(["epsilon_train_target", "epsilon_cal"]).to_string(index=False))

    dff = filter_df(
        df,
        # nominal_coverage=0.90,
        # coverage_target=0.89,
        # beta=1e-3,
        # cal_size=500,
    )

    # Plot figures.
    # plot_coverage_vs_formal_bound(dff, f"{analysis_out_dir}/coverage_vs_formal_bound.png")
    # plot_set_size_vs_epsilon_cal(dff, f"{analysis_out_dir}/set_size_vs_epsilon_cal.png")
    # plot_coverage_vs_epsilon_cal(dff, f"{analysis_out_dir}/coverage_vs_epsilon_cal.png")
    # plot_accuracy_vs_epsilon_train(dff, f"{analysis_out_dir}/accuracy_vs_epsilon_train.png")

    # Other versions gosh what a mess.
    plot_set_size_vs_epsilon_cal_by_nominal(
        df,
        f"{analysis_out_dir}/set_size_vs_epsilon_cal_by_nominal.png",
        fixed_cal_size=2000,
        fixed_epsilon_train_target=4.0,
        fixed_coverage_target=0.8,
    )

    plot_coverage_vs_epsilon_cal_by_nominal(
        df,
        f"{analysis_out_dir}/coverage_vs_epsilon_cal_by_nominal.png",
        fixed_cal_size=2000,
        fixed_epsilon_train_target=4.0,
        # fixed_coverage_target=0.7,
    )

    # Should have sharp boundaries.
    cal_size = None
    # cal_size = 500
    beta = 1e-3
    epsilon_cal = None
    # epsilon_cal = 0.1
    epsilon_train_target=None
    plot_contract_feasibility_map(
        df,
        f"{analysis_out_dir}/formal_contract_map_ecal_{epsilon_cal}_calsize_{cal_size}.png",
        value_col="overall_formal_ok",
        beta=beta,
        cal_size=cal_size,
        epsilon_train_target=epsilon_train_target,
        epsilon_cal=epsilon_cal,
        aggregate="all",
    )

    plot_contract_feasibility_map(
        df,
        f"{analysis_out_dir}/empirical_contract_map_ecal_{epsilon_cal}_calsize_{cal_size}.png",
        value_col="overall_empirical_ok",
        beta=beta,
        cal_size=cal_size,
        epsilon_train_target=epsilon_train_target,
        epsilon_cal=epsilon_cal,
        aggregate="mean",
        mask_formally_infeasible=False,
    )

    plot_contract_heatmap(
        df,
        f"{analysis_out_dir}/empirical_margin_to_target_map_ecal_{epsilon_cal}_calsize_{cal_size}.png",
        value_col="empirical_margin_to_target",
        beta=beta,
        cal_size=cal_size,
        epsilon_train_target=epsilon_train_target,
        epsilon_cal=epsilon_cal,
        aggregate="mean",
    )

    print(f"Saved outputs to {analysis_out_dir}")


if __name__ == "__main__":
    main()