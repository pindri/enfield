#!/usr/bin/env bash

python toy.py \
  --dataset cifar100 \
  --seed 0 \
  --epochs_np 15 \
  --epochs_dp 6 \
  --dp_eps_train 12.0 \
  --dp_eps_cal 8.0 \
  --cal_size 2000 \
  --nominal_coverage 0.8 \
  --coverage_target 0.8 \
  --beta 1e-3 \
  --batch_size 128 \
  --temperature 2.0 \
  --label_smoothing 0.0 \
  --verbose
