#!/usr/bin/env bash

  python toy.py \
  --dataset cifar10 \
  --seed 0 \
  --epochs_np 10 \
  --epochs_dp 4 \
  --dp_eps_train 4.0 \
  --dp_eps_cal 8.0 \
  --cal_size 2000 \
  --nominal_coverage 0.75 \
  --coverage_target 0.7 \
  --beta 1e-3 \
  --batch_size 128 \
  --label_smoothing 0.5 \
  --temperature 2.0 \
  --verbose \
  --out_dir toy_out_test