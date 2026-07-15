#!/usr/bin/env bash
# CIFAR10 main mechanism sweep: vary epsilon_cal, nominal_coverage, cal_size.
# Coverage target fixed at 0.70. Feeds analysis/main_mechanism_target070.

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
COVERAGE_TARGET=0.70

OUTDIR="toy_out_main_mechanism_target070"
ANALYSIS_DIR="analysis/main_mechanism_target070"

for seed in 0 1 2; do
  for cal_size in 1000 2000 4000; do
    for nomcov in 0.70 0.75 0.80 0.90; do
      for ceps in 1.0 2.0 4.0 8.0; do
        echo "[MAIN-MECH] seed=$seed cal_size=$cal_size nomcov=$nomcov ceps=$ceps"
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
          2>&1 | tee "logs/main_mech_seed${seed}_cal${cal_size}_nom${nomcov}_ceps${ceps}.log"
      done
    done
  done
done

python collect_reports.py --out_dir "$OUTDIR" --analysis_out_dir "$ANALYSIS_DIR"
