import argparse
from pathlib import Path

import pandas as pd

from plots import (
    plot_set_size_vs_epsilon_cal_by_nominal,
    plot_coverage_vs_epsilon_by_nominal,
    plot_certificate_vs_epsilon_by_cal_size,
    plot_inflation_vs_certificate,
    plot_method_tradeoff,
    save_theorem_summary_table,
    plot_contract_feasibility_map,
    plot_heatmap_choose,
    plot_accuracy_vs_epsilon_train,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_root", type=str, default="analysis")
    args = ap.parse_args()

    root = Path(args.analysis_root)

    # Input folders
    sectionA_dir = root / "main_mechanism_target070"
    req06_dir = root / "requirements_target06"
    req07_dir = root / "requirements_target07"
    req08_dir = root / "requirements_target08"
    train_priv_dir = root / "train_privacy_sensitivity_target070"
    saturation_dir = root / "saturation_check_target07"

    # Output folders
    out_main = root / "main"
    out_main.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    dfA = pd.read_csv(sectionA_dir / "all_reports.csv")
    dfR06 = pd.read_csv(req06_dir / "all_reports.csv")
    dfR07 = pd.read_csv(req07_dir / "all_reports.csv")
    dfR08 = pd.read_csv(req08_dir / "all_reports.csv")
    dfC = pd.read_csv(train_priv_dir / "all_reports.csv")
    dfD = pd.read_csv(saturation_dir / "all_reports.csv")

    # -----------------------------
    # Section A: main mechanism sweep
    # -----------------------------
    plot_set_size_vs_epsilon_cal_by_nominal(
        dfA,
        outpath=out_main / "sectionA_set_size_vs_epsilon_cal_by_nominal.png",
        aggregate="mean",
    )

    plot_coverage_vs_epsilon_by_nominal(
        dfA,
        outpath=out_main / "sectionA_coverage_vs_epsilon_cal_by_nominal.png",
        x_axis="epsilon_cal",
        aggregate="mean",
    )

    plot_certificate_vs_epsilon_by_cal_size(
        dfA,
        outpath=out_main / "sectionA_certificate_vs_epsilon_by_cal_size_nom075.png",
        fixed_nominal_coverage=0.75,
        aggregate="mean",
    )

    plot_inflation_vs_certificate(
        dfA,
        outpath=out_main / "sectionA_inflation_vs_certificate_cal2000_nom075.png",
        fixed_cal_size=2000,
        fixed_nominal_coverage=0.75,
    )

    plot_method_tradeoff(
        dfA,
        outpath=out_main / "sectionA_method_tradeoff_cal2000_nom075.png",
        fixed_cal_size=2000,
        fixed_nominal_coverage=0.75,
        aggregate="mean",
    )

    save_theorem_summary_table(
        dfA,
        outpath=out_main / "sectionA_theorem_summary.csv",
    )

    # -----------------------------
    # Requirements sweeps
    # -----------------------------
    plot_contract_feasibility_map(
        dfR06,
        outpath=out_main / "req06_formal_contract_map.png",
        value_col="overall_formal_ok",
        beta=1e-3,
        aggregate="all",
    )
    plot_contract_feasibility_map(
        dfR07,
        outpath=out_main / "req07_formal_contract_map.png",
        value_col="overall_formal_ok",
        beta=1e-3,
        aggregate="all",
    )
    plot_contract_feasibility_map(
        dfR08,
        outpath=out_main / "req08_formal_contract_map.png",
        value_col="overall_formal_ok",
        beta=1e-3,
        aggregate="all",
    )

    plot_contract_feasibility_map(
        dfR06,
        outpath=out_main / "req06_empirical_contract_map.png",
        value_col="overall_empirical_ok",
        beta=1e-3,
        aggregate="mean",
    )
    plot_contract_feasibility_map(
        dfR07,
        outpath=out_main / "req07_empirical_contract_map.png",
        value_col="overall_empirical_ok",
        beta=1e-3,
        aggregate="mean",
    )
    plot_contract_feasibility_map(
        dfR08,
        outpath=out_main / "req08_empirical_contract_map.png",
        value_col="overall_empirical_ok",
        beta=1e-3,
        aggregate="mean",
    )

    plot_heatmap_choose(
        dfR06,
        outpath=out_main / "req06_empirical_margin_to_target.png",
        value_col="empirical_margin_to_target",
        y_col="nominal_coverage",
        x_col="epsilon_cal",
        beta=1e-3,
        aggregate="mean",
        xlabel=r"$\epsilon_{\mathrm{cal}}$",
        ylabel="Nominal coverage",
        title="Empirical margin to target (target=0.6)",
    )
    plot_heatmap_choose(
        dfR07,
        outpath=out_main / "req07_empirical_margin_to_target.png",
        value_col="empirical_margin_to_target",
        y_col="nominal_coverage",
        x_col="epsilon_cal",
        beta=1e-3,
        aggregate="mean",
        xlabel=r"$\epsilon_{\mathrm{cal}}$",
        ylabel="Nominal coverage",
        title="Empirical margin to target (target=0.7)",
    )
    plot_heatmap_choose(
        dfR08,
        outpath=out_main / "req08_empirical_margin_to_target.png",
        value_col="empirical_margin_to_target",
        y_col="nominal_coverage",
        x_col="epsilon_cal",
        beta=1e-3,
        aggregate="mean",
        xlabel=r"$\epsilon_{\mathrm{cal}}$",
        ylabel="Nominal coverage",
        title="Empirical margin to target (target=0.8)",
    )

    # -----------------------------
    # Training privacy sensitivity
    # -----------------------------
    plot_coverage_vs_epsilon_by_nominal(
        dfC,
        outpath=out_main / "trainpriv_coverage_vs_epsilon_train_by_nominal.png",
        x_axis="epsilon_train_target",
        aggregate="mean",
    )

    plot_accuracy_vs_epsilon_train(
        dfC,
        outpath=out_main / "trainpriv_accuracy_vs_epsilon_train.png",
    )

    # -----------------------------
    # Saturation check
    # -----------------------------
    plot_heatmap_choose(
        dfD,
        outpath=out_main / "saturation_tau_dp_cal.png",
        value_col="tau_dp_cal",
        y_col="nominal_coverage",
        x_col="epsilon_cal",
        beta=1e-3,
        aggregate="mean",
        xlabel=r"$\epsilon_{\mathrm{cal}}$",
        ylabel="Nominal coverage",
        title=r"Mean $\tau_{\mathrm{dp}}$ in high-nominal regime",
    )

    plot_heatmap_choose(
        dfD,
        outpath=out_main / "saturation_empirical_minus_nominal.png",
        value_col="empirical_margin_to_nominal",
        y_col="nominal_coverage",
        x_col="epsilon_cal",
        beta=1e-3,
        aggregate="mean",
        xlabel=r"$\epsilon_{\mathrm{cal}}$",
        ylabel="Nominal coverage",
        title="Empirical coverage - nominal coverage (high-nominal regime)",
    )

    print(f"Saved plots and tables to {out_main}")


if __name__ == "__main__":
    main()
