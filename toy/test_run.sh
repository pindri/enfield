#!/usr/bin/env bash

python toy.py \
  --dataset cifar100 \
  --train_label_noise 0.0 \
  --epochs_np 10 \
  --epochs_dp 10 \
  --dp_train_eps 4.0 \
  --dp_eps_cal 8.0 \
  --nominal_coverage 0.5 \
  --coverage_target 0.5 \
  --cal_size 1000 \
  --temperature 2.0 \
  --label_smoothing 0.0
