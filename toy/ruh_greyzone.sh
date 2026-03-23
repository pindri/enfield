#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs


########################################
# Block D: Feasibility heatmap boundary
# Fixed train/cal privacy, vary nominal coverage and target
########################################

for seed in 0 1 2; do
  for nomcov in 0.7 0.8 0.9 0.93 0.95; do
    case "$nomcov" in
      0.7) targets="0.66 0.68 0.70 0.72" ;;
      0.8) targets="0.76 0.78 0.80 0.82" ;;
      0.9) targets="0.86 0.88 0.90 0.92" ;;
      0.93) targets="0.91 0.92 0.93 0.94 0.95" ;;
      0.95) targets="0.93 0.94 0.95 0.96 0.97" ;;
    esac

    for target in $targets; do
      echo "seed=$seed nomcov=$nomcov target=$target"
      python toy.py \
        --dataset cifar10 \
        --seed "$seed" \
        --epochs_np 10 \
        --epochs_dp 4 \
        --dp_train_eps 4.0 \
        --dp_eps_cal 4.0 \
        --cal_size 2000 \
        --nominal_coverage "$nomcov" \
        --coverage_target "$target" \
        --beta 1e-3 \
        --batch_size 128 \
        --temperature 2.0 \
        --verbose \
        --out_dir toy_out_blockD \
        2>&1 | tee "logs/blockD_seed${seed}_nom${nomcov}_target${target}.log"
    done
  done
done
