#!/bin/sh

# Base shared settings:
# - dataset=cifar10
# - epochs_np=10
# - epochs_dp=2
# - cal_size=2000
# - batch_size=128
# - temperature=2.0
# - beta=1e-3

########################################
# Calibration vs privacy
# Vary epsilon_cal, with target = nominal
########################################
for seed in 0 1 2; do
  for nomcov in 0.7 0.8 0.9; do
    for ceps in 1.0 2.0 4.0 8.0; do
      echo "A seed=$seed nomcov=$nomcov ceps=$ceps"
      python toy.py \
        --dataset cifar10 \
        --seed "$seed" \
        --epochs_np 10 \
        --epochs_dp 2 \
        --dp_train_eps 4.0 \
        --dp_cal_eps "$ceps" \
        --cal_size 2000 \
        --nominal_coverage "$nomcov" \
        --coverage_target "$nomcov" \
        --beta 1e-3 \
        --batch_size 128 \
        --temperature 2.0 \
        --verbose \
        2>&1 | tee "logs/blockA_seed${seed}_nom${nomcov}_ceps${ceps}.log"
    done
  done
done

########################################
# Training vs privacy
# Vary epsilon_train at fixed calibration setting
########################################
for seed in 0 1 2; do
  for teps in 2.0 4.0 8.0; do
    echo "B seed=$seed teps=$teps"
    python toy.py \
      --dataset cifar10 \
      --seed "$seed" \
      --epochs_np 10 \
      --epochs_dp 2 \
      --dp_train_eps "$teps" \
      --dp_cal_eps 4.0 \
      --cal_size 2000 \
      --nominal_coverage 0.8 \
      --coverage_target 0.8 \
      --beta 1e-3 \
      --batch_size 128 \
      --temperature 2.0 \
      --verbose \
      2>&1 | tee "logs/blockB_seed${seed}_teps${teps}.log"
  done
done

########################################
# Feasibility heatmap
# Fixed train/cal privacy, vary nominal coverage and target
########################################
for seed in 0 1 2; do
  for nomcov in 0.6 0.7 0.8 0.9; do
    for target in 0.6 0.7 0.8 0.9; do
      echo "C seed=$seed nomcov=$nomcov target=$target"
      python toy.py \
        --dataset cifar10 \
        --seed "$seed" \
        --epochs_np 10 \
        --epochs_dp 2 \
        --dp_train_eps 4.0 \
        --dp_cal_eps 4.0 \
        --cal_size 2000 \
        --nominal_coverage "$nomcov" \
        --coverage_target "$target" \
        --beta 1e-3 \
        --batch_size 128 \
        --temperature 2.0 \
        --verbose \
        2>&1 | tee "logs/blockC_seed${seed}_nom${nomcov}_target${target}.log"
    done
  done
done
