#!/usr/bin/env bash

set -euo pipefail
mkdir -p logs

# Base shared settings
DATASET="cifar10"
EPOCHS_NP=10
EPOCHS_DP=4
CAL_SIZE=2000
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.0

########################################
# Block A: Calibration vs privacy
# Vary epsilon_cal, with target = nominal
########################################
for seed in 0 1 2; do
  for nomcov in 0.7 0.8 0.9; do
    for ceps in 1.0 2.0 4.0 8.0; do
      echo "A seed=$seed nomcov=$nomcov ceps=$ceps"
      python toy.py \
        --dataset "$DATASET" \
        --seed "$seed" \
        --epochs_np "$EPOCHS_NP" \
        --epochs_dp "$EPOCHS_DP" \
        --dp_train_eps 4.0 \
        --dp_eps_cal "$ceps" \
        --cal_size "$CAL_SIZE" \
        --nominal_coverage "$nomcov" \
        --coverage_target "$nomcov" \
        --beta "$BETA" \
        --batch_size "$BATCH_SIZE" \
        --temperature "$TEMPERATURE" \
        --label_smoothing "$LABEL_SMOOTHING" \
        --verbose \
        --out_dir toy_out_blockA \
        2>&1 | tee "logs/blockA_seed${seed}_nom${nomcov}_ceps${ceps}.log"
    done
  done
done

########################################
# Block B: Training vs privacy
# Vary epsilon_train at fixed calibration setting
########################################
for seed in 0 1 2; do
  for teps in 1.0 2.0 4.0 8.0; do
    echo "B seed=$seed teps=$teps"
    python toy.py \
      --dataset "$DATASET" \
      --seed "$seed" \
      --epochs_np "$EPOCHS_NP" \
      --epochs_dp "$EPOCHS_DP" \
      --dp_train_eps "$teps" \
      --dp_eps_cal 4.0 \
      --cal_size "$CAL_SIZE" \
      --nominal_coverage 0.8 \
      --coverage_target 0.8 \
      --beta "$BETA" \
      --batch_size "$BATCH_SIZE" \
      --label_smoothing "$LABEL_SMOOTHING" \
      --temperature "$TEMPERATURE" \
      --verbose \
      --out_dir toy_out_blockB \
      2>&1 | tee "logs/blockB_seed${seed}_teps${teps}.log"
  done
done

########################################
# Block C: Feasibility heatmap
# Fixed train/cal privacy, vary nominal coverage and target
########################################
for seed in 0 1; do
  for nomcov in 0.6 0.7 0.8 0.9; do
    for target in 0.6 0.7 0.8 0.9; do
      echo "C seed=$seed nomcov=$nomcov target=$target"
      python toy.py \
        --dataset "$DATASET" \
        --seed "$seed" \
        --epochs_np "$EPOCHS_NP" \
        --epochs_dp "$EPOCHS_DP" \
        --dp_train_eps 4.0 \
        --dp_eps_cal 4.0 \
        --cal_size "$CAL_SIZE" \
        --nominal_coverage "$nomcov" \
        --coverage_target "$target" \
        --beta "$BETA" \
        --batch_size "$BATCH_SIZE" \
        --label_smoothing "$LABEL_SMOOTHING" \
        --temperature "$TEMPERATURE" \
        --verbose \
        --out_dir toy_out_blockC \
        2>&1 | tee "logs/blockC_seed${seed}_nom${nomcov}_target${target}.log"
    done
  done
done
