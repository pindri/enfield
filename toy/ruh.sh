#!/bin/sh

## 1) Calibration privacy sweep
#for seed in 0 1 2; do
#  for ceps in 1.0 2.0 4.0 8.0; do
#    python toy.py \
#      --dataset cifar10 \
#      --seed "$seed" \
#      --epochs_np 10 \
#      --epochs_dp 2 \
#      --dp_train_eps 4.0 \
#      --dp_cal_eps "$ceps" \
#      --cal_size 2000 \
#      --nominal_coverage 0.7 \
#      --coverage_target 0.7 \
#      --beta 1e-3 \
#      --batch_size 128 \
#      --temperature 2.0 \
#      --verbose
#  done
#done
#
## 2) Nominal coverage sweep
#for seed in 0 1 2; do
#  for nomcov in 0.6 0.7 0.8 0.9; do
#    python toy.py \
#      --dataset cifar10 \
#      --seed "$seed" \
#      --epochs_np 10 \
#      --epochs_dp 2 \
#      --dp_train_eps 4.0 \
#      --dp_cal_eps 4.0 \
#      --cal_size 2000 \
#      --nominal_coverage "$nomcov" \
#      --coverage_target 0.7 \
#      --beta 1e-3 \
#      --batch_size 128 \
#      --temperature 2.0 \
#      --verbose
#  done
#done
#
## 3) Small feasibility heatmap sweep
#for seed in 0 1 2; do
#  for nomcov in 0.6 0.7 0.8 0.9; do
#    for target in 0.6 0.7 0.8 0.9; do
#      python toy.py \
#        --dataset cifar10 \
#        --seed "$seed" \
#        --epochs_np 10 \
#        --epochs_dp 2 \
#        --dp_train_eps 4.0 \
#        --dp_cal_eps 4.0 \
#        --cal_size 2000 \
#        --nominal_coverage "$nomcov" \
#        --coverage_target "$target" \
#        --beta 1e-3 \
#        --batch_size 128 \
#        --temperature 2.0 \
#        --verbose
#    done
#  done
#done


for seed in 0 1 2; do
  for target in 0.6 0.8 0.9; do
    for ceps in 1.0 2.0 8.0; do
      echo "seed=$seed target=$target ceps=$ceps"
      python toy.py \
        --dataset cifar10 \
        --seed "$seed" \
        --epochs_np 10 \
        --epochs_dp 2 \
        --dp_train_eps 4.0 \
        --dp_cal_eps "$ceps" \
        --cal_size 2000 \
        --nominal_coverage 0.7 \
        --coverage_target "$target" \
        --beta 1e-3 \
        --batch_size 128 \
        --temperature 2.0 \
        --verbose
    done
  done
done