import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm

from utils import save_dataframe
from utils import ensure_dir, make_reproducible, load_reports


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

def plot_certificate_vs_epsilon_by_cal_size(
    df: pd.DataFrame,
    outpath: str | Path,
    fixed_nominal_coverage: float | None = None,
    aggregate: str = "mean",
) -> None:
    dff = df.copy()

    if fixed_nominal_coverage is not None:
        dff = dff[np.isclose(dff["nominal_coverage"], fixed_nominal_coverage)].copy()

    if dff.empty:
        raise ValueError("No rows remain after filtering for plot_certificate_vs_epsilon_by_cal_size")

    plt.figure(figsize=(7, 5))

    for cal_size in sorted(dff["cal_size"].unique()):
        g = dff[dff["cal_size"] == cal_size].copy()
        grouped = g.groupby("epsilon_cal", as_index=False)["certificate_width_tau"]

        if aggregate == "mean":
            gplot = grouped.mean()
        elif aggregate == "median":
            gplot = grouped.median()
        else:
            raise ValueError("aggregate must be 'mean' or 'median'")

        gplot = gplot.sort_values("epsilon_cal")

        plt.plot(
            gplot["epsilon_cal"],
            gplot["certificate_width_tau"],
            marker="o",
            label=f"cal_size={cal_size}",
        )

    plt.xlabel(r"$\epsilon_{\mathrm{cal}}$")
    plt.ylabel("Certificate width")
    plt.title("Certificate width vs calibration privacy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_theorem_summary_table(
    df: pd.DataFrame,
    outpath: str | Path,
) -> None:
    summary = (
        df.groupby(["epsilon_cal", "nominal_coverage", "cal_size"], as_index=False)
        .agg(
            certificate_width_tau_mean=("certificate_width_tau", "mean"),
            observed_inflation_tau_grid_mean=("observed_inflation_tau_grid", "mean"),
            avg_set_size_dp_cal_mean=("avg_set_size_dp_cal", "mean"),
            test_coverage_dp_cal_mean=("test_coverage_dp_cal", "mean"),
            theorem_tau_ok_rate=("theorem_tau_ok", "mean"),
            theorem_idx_ok_rate=("theorem_idx_ok", "mean"),
            inflation_to_certificate_ratio_mean=("inflation_to_certificate_ratio", "mean"),
        )
    )
    summary.to_csv(outpath, index=False)

def plot_method_tradeoff( df: pd.DataFrame, outpath: str | Path, fixed_cal_size: int | None = None,
                          fixed_nominal_coverage: float | None = None, aggregate: str = "mean", ) -> None:
    dff = df.copy()

    if fixed_cal_size is not None:
        dff = dff[dff["cal_size"] == fixed_cal_size].copy()
    if fixed_nominal_coverage is not None:
        dff = dff[np.isclose(dff["nominal_coverage"], fixed_nominal_coverage)].copy()

    if dff.empty:
        raise ValueError("No rows remain after filtering for plot_method_tradeoff")

    if aggregate == "mean":
        agg_fn = "mean"
    elif aggregate == "median":
        agg_fn = "median"
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")

    np_cov = getattr(dff["np_coverage"], agg_fn)()
    np_size = getattr(dff["np_avg_set_size"], agg_fn)()

    dp_np_cov = getattr(dff["test_coverage_nonprivate_cal"], agg_fn)()
    dp_np_size = getattr(dff["dp_avg_set_size_nonprivate_cal"], agg_fn)()

    dp_dp_cov = getattr(dff["test_coverage_dp_cal"], agg_fn)()
    dp_dp_size = getattr(dff["avg_set_size_dp_cal"], agg_fn)()

    plt.figure(figsize=(6, 5))
    plt.scatter([np_size], [np_cov], s=80, label="NP model + NP cal")
    plt.scatter([dp_np_size], [dp_np_cov], s=80, label="DP model + NP cal")
    plt.scatter([dp_dp_size], [dp_dp_cov], s=80, label="DP model + DP cal")

    plt.xlabel("Average prediction set size")
    plt.ylabel("Empirical coverage")
    plt.title("Coverage vs set size by method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_inflation_vs_certificate(
    df: pd.DataFrame,
    outpath: str | Path,
    fixed_cal_size: int | None = None,
    fixed_nominal_coverage: float | None = None,
) -> None:
    dff = df.copy()

    if fixed_cal_size is not None:
        dff = dff[dff["cal_size"] == fixed_cal_size].copy()
    if fixed_nominal_coverage is not None:
        dff = dff[np.isclose(dff["nominal_coverage"], fixed_nominal_coverage)].copy()

    if dff.empty:
        raise ValueError("No rows remain after filtering for plot_inflation_vs_certificate")

    plt.figure(figsize=(6, 5))

    for eps_cal in sorted(dff["epsilon_cal"].unique()):
        g = dff[np.isclose(dff["epsilon_cal"], eps_cal)]
        plt.scatter(
            g["certificate_width_tau"],
            g["observed_inflation_tau_grid"],
            label=f"eps_cal={eps_cal}",
        )

    hi = max(
        float(dff["certificate_width_tau"].max()),
        float(dff["observed_inflation_tau_grid"].max()),
    )
    plt.plot([0, hi], [0, hi], linestyle="--")
    plt.xlabel("Certificate width")
    plt.ylabel("Observed inflation (grid)")
    plt.title("Observed inflation vs certificate width")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

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


def plot_heatmap_choose(
    df: pd.DataFrame,
    outpath: str | Path,
    value_col: str,
    y_col: str,
    x_col: str,
    beta: float | None = None,
    cal_size: int | None = None,
    epsilon_train_target: float | None = None,
    epsilon_cal: float | None = None,
    aggregate: str = "mean",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    dff = df.copy()

    if beta is not None:
        dff = dff[np.isclose(dff["beta"], beta)]
    if cal_size is not None:
        dff = dff[dff["cal_size"] == cal_size]
    if epsilon_train_target is not None:
        dff = dff[np.isclose(dff["epsilon_train_target"], epsilon_train_target)]
    if epsilon_cal is not None:
        dff = dff[np.isclose(dff["epsilon_cal"], epsilon_cal)]

    if dff.empty:
        raise ValueError("No rows remain after filtering")

    grouped = dff.groupby([y_col, x_col], as_index=False)[value_col]

    if aggregate == "mean":
        agg = grouped.mean()
    elif aggregate == "median":
        agg = grouped.median()
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")

    pivot = agg.pivot(
        index=y_col,
        columns=x_col,
        values=value_col,
    ).sort_index().sort_index(axis=1)

    plot_data = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(7, 5))

    finite_vals = plot_data[np.isfinite(plot_data)]
    if finite_vals.size == 0:
        raise ValueError("No finite values to plot")

    vmin = min(0.0, float(np.nanmin(finite_vals)))
    vmax = max(0.0, float(np.nanmax(finite_vals)))

    if np.isclose(vmin, vmax):
        im = plt.imshow(
            plot_data,
            aspect="auto",
            interpolation="nearest",
            cmap="RdYlGn",
            vmin=vmin - 1e-8,
            vmax=vmax + 1e-8,
        )
    else:
        norm = TwoSlopeNorm(vmin=vmin - 1e-10, vcenter=0.0, vmax=vmax + 1e-10)
        im = plt.imshow(
            plot_data,
            aspect="auto",
            interpolation="nearest",
            cmap="RdYlGn",
            norm=norm,
        )

    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(y) for y in pivot.index])

    plt.xlabel(xlabel if xlabel is not None else x_col)
    plt.ylabel(ylabel if ylabel is not None else y_col)
    plt.title(title if title is not None else value_col)

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


def plot_coverage_vs_epsilon_by_nominal(
    df: pd.DataFrame,
    outpath: str | Path,
    fixed_cal_size: int | None = None,
    fixed_epsilon_train_target: float | None = None,
    fixed_coverage_target: float | None = None,
    fixed_nominal_coverage: float | None = None,
    aggregate: str = "mean",
    x_axis: str = "epsilon_cal",
) -> None:
    """
    Line plot of empirical coverage vs selected epsilon axis, with one curve per nominal coverage.
    If multiple rows remain per x value (e.g. multiple seeds), aggregate them.

    x_axis can be one of: 'epsilon_cal', 'epsilon_train_target', 'epsilon_train_realized'.
    """
    dff = df.copy()

    axis_meta = {
        "epsilon_cal": (
            r"$\epsilon_{\mathrm{cal}}$",
            "Coverage vs calibration privacy",
        ),
        "epsilon_train_target": (
            r"Target $\epsilon_{\mathrm{train}}$",
            "Coverage vs target training privacy",
        ),
        "epsilon_train_realized": (
            r"Realized $\epsilon_{\mathrm{train}}$",
            "Coverage vs realized training privacy",
        ),
    }
    if x_axis not in axis_meta:
        valid = ", ".join(axis_meta.keys())
        raise ValueError(f"x_axis must be one of: {valid}")

    if fixed_cal_size is not None:
        dff = dff[dff["cal_size"] == fixed_cal_size].copy()

    if fixed_epsilon_train_target is not None:
        dff = dff[np.isclose(dff["epsilon_train_target"], fixed_epsilon_train_target)].copy()

    if fixed_coverage_target is not None:
        dff = dff[np.isclose(dff["coverage_target"], fixed_coverage_target)].copy()

    if fixed_nominal_coverage is not None:
        dff = dff[np.isclose(dff["nominal_coverage"], fixed_nominal_coverage)].copy()

    if dff.empty:
        raise ValueError("No rows remain after filtering for plot_coverage_vs_epsilon_cal_by_nominal")

    plt.figure(figsize=(7, 5))

    for nomcov in sorted(dff["nominal_coverage"].unique()):
        g = dff[np.isclose(dff["nominal_coverage"], nomcov)].copy()

        grouped = g.groupby(x_axis, as_index=False)["test_coverage_dp_cal"]
        if aggregate == "mean":
            gplot = grouped.mean()
        elif aggregate == "median":
            gplot = grouped.median()
        else:
            raise ValueError("aggregate must be 'mean' or 'median'")

        gplot = gplot.sort_values(x_axis)

        plt.plot(
            gplot[x_axis],
            gplot["test_coverage_dp_cal"],
            marker="o",
            label=f"nom={nomcov}",
        )

    xlabel, title = axis_meta[x_axis]
    plt.xlabel(xlabel)
    plt.ylabel("Empirical coverage")
    plt.title(title)
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

    pivot = pivot.astype(float)

    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest",
               vmin=0, vmax=1.0, cmap="viridis"
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
    ap.add_argument("--analysis_out_dir", type=str, default="analysis/main")
    # ap.add_argument("--analysis_out_dir", type=str, default="analysis_smoothckeck")
    # ap.add_argument("--out_dir", type=str, default="toy_out")
    # ap.add_argument("--out_dir", type=str, default="toy_out_cifar10/toy_out_blockC")
    ap.add_argument("--out_dir", type=str, default="toy_out_main")
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


    # Other versions gosh what a mess.
    plot_set_size_vs_epsilon_cal_by_nominal(
        df,
        f"{analysis_out_dir}/set_size_vs_epsilon_cal_by_nominal.png",
        fixed_cal_size=2000,
        fixed_epsilon_train_target=4.0,
        # fixed_coverage_target=0.8,
    )

    plot_coverage_vs_epsilon_by_nominal(
        df,
        f"{analysis_out_dir}/coverage_vs_epsilon_cal_by_nominal.png",
        fixed_cal_size=2000,
        fixed_epsilon_train_target=4.0,
        x_axis="epsilon_cal",
        # fixed_nominal_coverage=0.8,
    )

    plot_coverage_vs_epsilon_by_nominal(
        df,
        f"{analysis_out_dir}/coverage_vs_epsilon_train_by_nominal.png",
        fixed_cal_size=2000,
        # fixed_epsilon_train_target=4.0,
        x_axis="epsilon_train_target",
        # fixed_nominal_coverage=0.8,
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

    plot_heatmap_choose(
        df,
        outpath=f"{analysis_out_dir}/heatmap_empirical_margin_to_target.png",
        value_col="empirical_margin_to_target",
        y_col="nominal_coverage",
        x_col="coverage_target",
        beta=1e-3,
        cal_size=2000,
        epsilon_train_target=4.0,
        epsilon_cal=4.0,
        aggregate="mean",
        xlabel="Coverage target",
        ylabel="Nominal coverage",
        title="Empirical margin to target",
    )

    plot_heatmap_choose(
        df,
        outpath=f"{analysis_out_dir}/heatmap_empirical_minus_nominal.png",
        value_col="empirical_margin_to_nominal",
        y_col="nominal_coverage",
        x_col="epsilon_cal",
        beta=1e-3,
        cal_size=2000,
        epsilon_train_target=4.0,
        aggregate="mean",
        xlabel=r"$\epsilon_{\mathrm{cal}}$",
        ylabel="Nominal coverage",
        title="Empirical coverage - nominal coverage",
    )
    print(f"Saved outputs to {analysis_out_dir}")

    # NEW PLOTS.
    plot_inflation_vs_certificate(
        df,
        f"{analysis_out_dir}/inflation_vs_certificate_cal2000_nom075.png",
        fixed_cal_size=2000,
        fixed_nominal_coverage=0.75,
    )

    plot_certificate_vs_epsilon_by_cal_size(
        df,
        f"{analysis_out_dir}/certificate_vs_epsilon_by_cal_size_nom075.png",
        fixed_nominal_coverage=0.75,
    )

    plot_method_tradeoff(
        df,
        f"{analysis_out_dir}/method_tradeoff_cal2000_nom075.png",
        fixed_cal_size=2000,
        fixed_nominal_coverage=0.75,
    )

    save_theorem_summary_table(
        df,
        f"{analysis_out_dir}/theorem_summary.csv",
    )


if __name__ == "__main__":
    main()