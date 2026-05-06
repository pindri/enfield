#!/usr/bin/env bash

set -euo pipefail
mkdir -p logs

# Base shared settings
DATASET="cifar100"
EPOCHS_NP=10
EPOCHS_DP=4
CAL_SIZE=2000
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.5

########################################
# Block C: Feasibility heatmap
# Fixed train/cal privacy, vary nominal coverage and target
########################################
for seed in 0 1; do
  for nomcov in 0.45 0.55 0.65 0.75 0.85 0.95; do
    for target in 0.4 0.5 0.6 0.7 0.8 0.9; do
      echo "C seed=$seed nomcov=$nomcov target=$target"
      python toy.py \
        --dataset "$DATASET" \
        --seed "$seed" \
        --epochs_np "$EPOCHS_NP" \
        --epochs_dp "$EPOCHS_DP" \
        --dp_eps_train 4.0 \
        --dp_eps_cal 4.0 \
        --cal_size "$CAL_SIZE" \
        --nominal_coverage "$nomcov" \
        --coverage_target "$target" \
        --beta "$BETA" \
        --batch_size "$BATCH_SIZE" \
        --label_smoothing "$LABEL_SMOOTHING" \
        --temperature "$TEMPERATURE" \
        --train_label_noise 0.0 \
        --verbose \
        --out_dir toy_out_blockC \
        2>&1 | tee "logs/blockC_seed${seed}_nom${nomcov}_target${target}.log"
    done
  done
done
