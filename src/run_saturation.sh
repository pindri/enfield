#!/usr/bin/env bash
# CIFAR10 saturation check: sweep epsilon_cal and nominal_coverage in the high-nominal regime.
# Checks whether tau_dp saturates and how empirical coverage deviates from nominal.
# Coverage target fixed at 0.70. Feeds analysis/saturation_check_target07.

set -euo pipefail
mkdir -p logs

DATASET="cifar10"
EPOCHS_NP=10
EPOCHS_DP=4
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.0
DP_EPS_TRAIN=4.0
CAL_SIZE=2000
COVERAGE_TARGET=0.70

OUTDIR="toy_out_saturation_check_target07"
ANALYSIS_DIR="analysis/saturation_check_target07"

for seed in 0 1; do
  for nomcov in 0.80 0.85 0.90 0.95; do
    for ceps in 1.0 2.0 4.0 8.0; do
      echo "[SATURATION] seed=$seed nomcov=$nomcov ceps=$ceps"
      python toy.py \
        --dataset "$DATASET" \
        --seed "$seed" \
        --epochs_np "$EPOCHS_NP" \
        --epochs_dp "$EPOCHS_DP" \
        --dp_eps_train "$DP_EPS_TRAIN" \
        --dp_eps_cal "$ceps" \
        --cal_size "$CAL_SIZE" \
        --nominal_coverage "$nomcov" \
        --coverage_target "$COVERAGE_TARGET" \
        --beta "$BETA" \
        --batch_size "$BATCH_SIZE" \
        --temperature "$TEMPERATURE" \
        --label_smoothing "$LABEL_SMOOTHING" \
        --verbose \
        --out_dir "$OUTDIR" \
        2>&1 | tee "logs/saturation_seed${seed}_nom${nomcov}_ceps${ceps}.log"
    done
  done
done

python collect_reports.py --out_dir "$OUTDIR" --analysis_out_dir "$ANALYSIS_DIR"
