#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

DATASET="fashionmnist"
OUTDIR_A="toy_out_fmnist_main_mechanism_target070"

EPOCHS_NP=10
EPOCHS_DP=4
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.5
DP_EPS_TRAIN=4.0

for seed in 0 1 2; do
  for cal_size in 1000 2000 4000; do
    for nomcov in 0.55 0.65 0.75; do
      for ceps in 2.0 4.0 8.0; do
        echo "[FMNIST-A] seed=$seed cal_size=$cal_size nomcov=$nomcov ceps=$ceps"
        python toy.py \
          --dataset "$DATASET" \
          --seed "$seed" \
          --epochs_np "$EPOCHS_NP" \
          --epochs_dp "$EPOCHS_DP" \
          --dp_eps_train "$DP_EPS_TRAIN" \
          --dp_eps_cal "$ceps" \
          --cal_size "$cal_size" \
          --nominal_coverage "$nomcov" \
          --coverage_target 0.7 \
          --beta "$BETA" \
          --batch_size "$BATCH_SIZE" \
          --temperature "$TEMPERATURE" \
          --label_smoothing "$LABEL_SMOOTHING" \
          --verbose \
          --out_dir "$OUTDIR_A" \
          2>&1 | tee "logs/fmnist_main_seed${seed}_cal${cal_size}_nom${nomcov}_ceps${ceps}.log"
      done
    done
  done
done
