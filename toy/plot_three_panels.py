#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summarize_curve(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    aggregate: str = "mean",
) -> pd.DataFrame:
    grouped = (
        df.groupby(x_col)[y_col]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values(x_col)
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    if aggregate == "mean":
        grouped["center"] = grouped["mean"]
    elif aggregate == "median":
        grouped["center"] = grouped["median"]
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")

    grouped["lower_1std"] = grouped["center"] - grouped["std"]
    grouped["upper_1std"] = grouped["center"] + grouped["std"]
    return grouped


def make_three_panel_figure(
    df: pd.DataFrame,
    outpath: str | Path,
    aggregate: str = "mean",
) -> None:

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    palette1 = sns.color_palette("colorblind", n_colors=len(sorted(df["cal_size"].unique())))
    palette2 = sns.color_palette("colorblind", n_colors=len(sorted(df["nominal_coverage"].unique())))
    palette3 = sns.color_palette("colorblind", n_colors=len(sorted(df["epsilon_cal"].unique())))


    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # -----------------------------
    # Panel 1: certificate width vs epsilon_cal
    # fixed nominal_coverage = 0.75
    # one curve per cal_size
    # -----------------------------
    ax = axes[0]
    dff = df[np.isclose(df["nominal_coverage"], 0.75)].copy()

    if dff.empty:
        raise ValueError("No rows remain for panel 1 after filtering nominal_coverage=0.75")

    # for cal_size in sorted(dff["cal_size"].unique()):
    for color, cal_size in zip(palette1, sorted(dff["cal_size"].unique())):
        g = dff[dff["cal_size"] == cal_size].copy()
        gplot = summarize_curve(
            g,
            x_col="epsilon_cal",
            y_col="certificate_width_tau",
            aggregate=aggregate,
        )

        ax.plot(
            gplot["epsilon_cal"],
            gplot["center"],
            marker="o",
            color=color,
            label=f"cal={cal_size}",
        )
        ax.fill_between(
            gplot["epsilon_cal"],
            gplot["lower_1std"],
            gplot["upper_1std"],
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel(r"$\epsilon_{\mathrm{cal}}$")
    ax.set_ylabel("Certificate width")
    ax.set_title("Certificate width vs calibration privacy")
    ax.legend()

    # -----------------------------
    # Panel 2: set size vs epsilon_cal
    # one curve per nominal_coverage
    # -----------------------------
    ax = axes[1]
    dff = df.copy()

    if dff.empty:
        raise ValueError("No rows remain for panel 2")

    # for nomcov in sorted(dff["nominal_coverage"].unique()):
    for color, nomcov in zip(palette2, sorted(dff["nominal_coverage"].unique())):
        g = dff[np.isclose(dff["nominal_coverage"], nomcov)].copy()
        gplot = summarize_curve(
            g,
            x_col="epsilon_cal",
            y_col="avg_set_size_dp_cal",
            aggregate=aggregate,
        )

        ax.plot(
            gplot["epsilon_cal"],
            gplot["center"],
            marker="o",
            color=color,
            label=f"nom={nomcov}",
        )
        ax.fill_between(
            gplot["epsilon_cal"],
            gplot["lower_1std"],
            gplot["upper_1std"],
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel(r"$\epsilon_{\mathrm{cal}}$")
    ax.set_ylabel("Average prediction set size")
    ax.set_title("Set size vs calibration privacy")
    ax.legend()

    # -----------------------------
    # Panel 3: observed inflation vs certificate width
    # fixed cal_size=2000, nominal_coverage=0.75
    # colored by epsilon_cal
    # -----------------------------
    ax = axes[2]
    dff = df[
        (df["cal_size"] == 2000) &
        (np.isclose(df["nominal_coverage"], 0.75))
    ].copy()

    if dff.empty:
        raise ValueError("No rows remain for panel 3 after filtering cal_size=2000 and nominal_coverage=0.75")
    for color, eps_cal in zip(palette3, sorted(dff["epsilon_cal"].unique())):
    # for eps_cal in sorted(dff["epsilon_cal"].unique()):
        g = dff[np.isclose(dff["epsilon_cal"], eps_cal)].copy()
        ax.scatter(
            g["certificate_width_tau"],
            g["observed_inflation_tau_grid"],
            color=color,
            label=f"eps={eps_cal}",
        )

    hi = max(
        float(dff["certificate_width_tau"].max()),
        float(dff["observed_inflation_tau_grid"].max()),
    )
    ax.plot([0, hi], [0, hi], linestyle="--")
    ax.set_xlabel("Certificate width")
    ax.set_ylabel("Observed inflation")
    ax.set_title("Observed inflation vs certificate")
    ax.legend()

    plt.tight_layout()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="analysis/main_mechanism_target070/all_reports.csv",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="analysis/main/three_panel_main_results.png",
    )
    ap.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "median"],
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    make_three_panel_figure(df, args.out, aggregate=args.aggregate)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()