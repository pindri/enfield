#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs

DATASET="cifar100"
EPOCHS_NP=10
EPOCHS_DP=4
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.5
OUTDIR="toy_out_cert"

for seed in 0 1; do
  for cal_size in 1000 2000 4000; do
    for eps_cal in 1.0 2.0 4.0 8.0; do
      for nomcov in 0.55 0.75 0.95; do
        echo "seed=$seed cal_size=$cal_size eps_cal=$eps_cal nomcov=$nomcov"
        python toy.py \
          --dataset "$DATASET" \
          --seed "$seed" \
          --epochs_np "$EPOCHS_NP" \
          --epochs_dp "$EPOCHS_DP" \
          --dp_eps_train 4.0 \
          --dp_eps_cal "$eps_cal" \
          --cal_size "$cal_size" \
          --nominal_coverage "$nomcov" \
          --coverage_target 0.7 \
          --beta "$BETA" \
          --batch_size "$BATCH_SIZE" \
          --label_smoothing "$LABEL_SMOOTHING" \
          --temperature "$TEMPERATURE" \
          --train_label_noise 0.0 \
          --verbose \
          --out_dir "$OUTDIR" \
          2>&1 | tee "logs/cert_seed${seed}_cal${cal_size}_eps${eps_cal}_nom${nomcov}.log"
      done
    done
  done
done
