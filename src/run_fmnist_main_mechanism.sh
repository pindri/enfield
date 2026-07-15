#!/usr/bin/env bash
# FashionMNIST main mechanism sweep: vary epsilon_cal, nominal_coverage, cal_size.
# Coverage target fixed at 0.70. Feeds analysis/fmnist_main_mechanism_target070.
# Use plot_three_panels.py to generate plots from the resulting CSV.

set -euo pipefail
mkdir -p logs

DATASET="fashionmnist"
EPOCHS_NP=10
EPOCHS_DP=4
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.5
DP_EPS_TRAIN=4.0
COVERAGE_TARGET=0.70

OUTDIR="toy_out_fmnist_main_mechanism_target070"
ANALYSIS_DIR="analysis/fmnist_main_mechanism_target070"

for seed in 0 1 2; do
  for cal_size in 1000 2000 4000; do
    for nomcov in 0.70 0.75 0.80; do
      for ceps in 2.0 4.0 8.0; do
        echo "[FMNIST-MAIN-MECH] seed=$seed cal_size=$cal_size nomcov=$nomcov ceps=$ceps"
        python toy.py \
          --dataset "$DATASET" \
          --seed "$seed" \
          --epochs_np "$EPOCHS_NP" \
          --epochs_dp "$EPOCHS_DP" \
          --dp_eps_train "$DP_EPS_TRAIN" \
          --dp_eps_cal "$ceps" \
          --cal_size "$cal_size" \
          --nominal_coverage "$nomcov" \
          --coverage_target "$COVERAGE_TARGET" \
          --beta "$BETA" \
          --batch_size "$BATCH_SIZE" \
          --temperature "$TEMPERATURE" \
          --label_smoothing "$LABEL_SMOOTHING" \
          --verbose \
          --out_dir "$OUTDIR" \
          2>&1 | tee "logs/fmnist_main_mech_seed${seed}_cal${cal_size}_nom${nomcov}_ceps${ceps}.log"
      done
    done
  done
done

python collect_reports.py --out_dir "$OUTDIR" --analysis_out_dir "$ANALYSIS_DIR"

python plot_three_panels.py \
  --csv "$ANALYSIS_DIR/all_reports.csv" \
  --out "$ANALYSIS_DIR/three_panel_main_results.pdf"
