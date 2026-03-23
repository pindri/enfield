#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

for seed in 0 1 2; do
  for smooth in 0.0 0.1; do
    for ceps in 4.0 8.0; do
      echo "seed=$seed smooth=$smooth ceps=$ceps"
      python toy.py \
        --dataset cifar10 \
        --seed "$seed" \
        --epochs_np 10 \
        --epochs_dp 4 \
        --dp_train_eps 4.0 \
        --dp_eps_cal "$ceps" \
        --cal_size 2000 \
        --nominal_coverage 0.8 \
        --coverage_target 0.8 \
        --beta 1e-3 \
        --batch_size 128 \
        --temperature 2.0 \
        --label_smoothing "$smooth" \
        --out_dir "toy_out_smoothcheck" \
        --verbose \
        2>&1 | tee "logs/smoothcheck_seed${seed}_smooth${smooth}_ceps${ceps}.log"
    done
  done
done
