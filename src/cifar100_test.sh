#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

DATASET="cifar100"
OUTDIR="toy_out_cifar100_diagnostic"
ANALYSIS_DIR="analysis/cifar100_diagnostic"

EPOCHS_NP=20
EPOCHS_DP=20
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.5
DP_EPS_TRAIN=4.0
COVERAGE_TARGET=0.7

for seed in 0 1; do
  for cal_size in 2000 4000; do
    for nomcov in 0.75; do
      for ceps in 4.0 8.0; do
        echo "[CIFAR100-DIAG] seed=$seed cal_size=$cal_size nomcov=$nomcov ceps=$ceps"
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
          2>&1 | tee "logs/cifar100_diag_seed${seed}_cal${cal_size}_nom${nomcov}_ceps${ceps}.log"
      done
    done
  done
done

python collect_reports.py --out_dir "$OUTDIR" --analysis_out_dir "$ANALYSIS_DIR"