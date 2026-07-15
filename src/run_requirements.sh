#!/usr/bin/env bash
# CIFAR10 requirements feasibility sweep: vary epsilon_cal and nominal_coverage
# for three fixed coverage targets (0.6, 0.7, 0.8). Feeds analysis/requirements_target0{6,7,8}.

set -euo pipefail
mkdir -p logs

DATASET="cifar10"
EPOCHS_NP=10
EPOCHS_DP=4
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.0
DP_EPS_TRAIN=4.0
CAL_SIZE=2000

for target in 0.6 0.7 0.8; do
  target_tag=$(echo "$target" | tr -d '.')
  OUTDIR="toy_out_requirements_target${target_tag}"
  ANALYSIS_DIR="analysis/requirements_target${target_tag}"

  for seed in 0 1; do
    for nomcov in 0.60 0.70 0.80 0.90; do
      for ceps in 1.0 2.0 4.0 8.0; do
        echo "[REQUIREMENTS] target=$target seed=$seed nomcov=$nomcov ceps=$ceps"
        python toy.py \
          --dataset "$DATASET" \
          --seed "$seed" \
          --epochs_np "$EPOCHS_NP" \
          --epochs_dp "$EPOCHS_DP" \
          --dp_eps_train "$DP_EPS_TRAIN" \
          --dp_eps_cal "$ceps" \
          --cal_size "$CAL_SIZE" \
          --nominal_coverage "$nomcov" \
          --coverage_target "$target" \
          --beta "$BETA" \
          --batch_size "$BATCH_SIZE" \
          --temperature "$TEMPERATURE" \
          --label_smoothing "$LABEL_SMOOTHING" \
          --verbose \
          --out_dir "$OUTDIR" \
          2>&1 | tee "logs/req_target${target_tag}_seed${seed}_nom${nomcov}_ceps${ceps}.log"
      done
    done
  done

  python collect_reports.py --out_dir "$OUTDIR" --analysis_out_dir "$ANALYSIS_DIR"
done
