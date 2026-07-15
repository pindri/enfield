#!/usr/bin/env bash
# CIFAR10 training-privacy sensitivity sweep: vary epsilon_train at fixed calibration settings.
# Coverage target fixed at 0.70. Feeds analysis/train_privacy_sensitivity_target070.

set -euo pipefail
mkdir -p logs

DATASET="cifar10"
EPOCHS_NP=10
EPOCHS_DP=4
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.0
DP_EPS_CAL=4.0
CAL_SIZE=2000
COVERAGE_TARGET=0.70

OUTDIR="toy_out_train_privacy_sensitivity_target070"
ANALYSIS_DIR="analysis/train_privacy_sensitivity_target070"

for seed in 0 1 2; do
  for teps in 1.0 2.0 4.0 8.0; do
    for nomcov in 0.70 0.75 0.80 0.90; do
      echo "[TRAIN-PRIV] seed=$seed teps=$teps nomcov=$nomcov"
      python toy.py \
        --dataset "$DATASET" \
        --seed "$seed" \
        --epochs_np "$EPOCHS_NP" \
        --epochs_dp "$EPOCHS_DP" \
        --dp_eps_train "$teps" \
        --dp_eps_cal "$DP_EPS_CAL" \
        --cal_size "$CAL_SIZE" \
        --nominal_coverage "$nomcov" \
        --coverage_target "$COVERAGE_TARGET" \
        --beta "$BETA" \
        --batch_size "$BATCH_SIZE" \
        --temperature "$TEMPERATURE" \
        --label_smoothing "$LABEL_SMOOTHING" \
        --verbose \
        --out_dir "$OUTDIR" \
        2>&1 | tee "logs/trainpriv_seed${seed}_teps${teps}_nom${nomcov}.log"
    done
  done
done

python collect_reports.py --out_dir "$OUTDIR" --analysis_out_dir "$ANALYSIS_DIR"
