#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

########################################
# Shared settings
########################################
DATASET="cifar10"
EPOCHS_NP=10
EPOCHS_DP=4
BATCH_SIZE=128
TEMPERATURE=2.0
BETA=1e-3
LABEL_SMOOTHING=0.5
DP_EPS_TRAIN=4.0

########################################
# Section A: Main mechanism sweep
# Purpose:
# - theorem validation
# - utility/privacy tradeoff
# - baseline plots
#
# Sweep:
#   eps_cal in {2,4,8}
#   nominal_coverage in {0.55,0.65,0.75}
#   cal_size in {1000,2000,4000}
#   seeds in {0,1,2}
#   coverage_target fixed at 0.7
########################################
#OUTDIR_A="toy_out_main_mechanism_target070"
#
#for seed in 0 1 2; do
#  for cal_size in 1000 2000 4000; do
#    for nomcov in 0.55 0.65 0.75; do
#      for ceps in 2.0 4.0 8.0; do
#        echo "[A] seed=$seed cal_size=$cal_size nomcov=$nomcov ceps=$ceps"
#        python toy.py \
#          --dataset "$DATASET" \
#          --seed "$seed" \
#          --epochs_np "$EPOCHS_NP" \
#          --epochs_dp "$EPOCHS_DP" \
#          --dp_eps_train "$DP_EPS_TRAIN" \
#          --dp_eps_cal "$ceps" \
#          --cal_size "$cal_size" \
#          --nominal_coverage "$nomcov" \
#          --coverage_target 0.7 \
#          --beta "$BETA" \
#          --batch_size "$BATCH_SIZE" \
#          --temperature "$TEMPERATURE" \
#          --label_smoothing "$LABEL_SMOOTHING" \
#          --verbose \
#          --out_dir "$OUTDIR_A" \
#          2>&1 | tee "logs/A_main_seed${seed}_cal${cal_size}_nom${nomcov}_ceps${ceps}.log"
#      done
#    done
#  done
#done
#

########################################
# Section B: Requirement-to-solution sweep
# Purpose:
# - isolate effect of changing requested coverage target
# - supports compiler/cards story
#
# Sweep:
#   coverage_target in {0.6,0.7,0.8}
#   nominal_coverage in {0.55,0.65,0.75,0.85}
#   eps_cal in {2,4,8}
#   cal_size in {1000,2000,4000}
#   seeds in {0,1,2}
########################################
for target in 0.6 0.7 0.8; do
  target_tag="${target/./}"
  OUTDIR_B="toy_out_requirements_target${target_tag}"
#  OUTDIR_B="toy_out_requirements_target$(echo "$target" | tr '.' '')"

  for seed in 0 1 2; do
    for cal_size in 1000 2000 4000; do
      for nomcov in 0.55 0.65 0.75 0.85; do
        for ceps in 2.0 4.0 8.0; do
          echo "[B] target=$target seed=$seed cal_size=$cal_size nomcov=$nomcov ceps=$ceps"
          python toy.py \
            --dataset "$DATASET" \
            --seed "$seed" \
            --epochs_np "$EPOCHS_NP" \
            --epochs_dp "$EPOCHS_DP" \
            --dp_eps_train "$DP_EPS_TRAIN" \
            --dp_eps_cal "$ceps" \
            --cal_size "$cal_size" \
            --nominal_coverage "$nomcov" \
            --coverage_target "$target" \
            --beta "$BETA" \
            --batch_size "$BATCH_SIZE" \
            --temperature "$TEMPERATURE" \
            --label_smoothing "$LABEL_SMOOTHING" \
            --verbose \
            --out_dir "$OUTDIR_B" \
            2>&1 | tee "logs/B_req_target${target}_seed${seed}_cal${cal_size}_nom${nomcov}_ceps${ceps}.log"
        done
      done
    done
  done
done

########################################
# Section C: Training-privacy sensitivity
# Purpose:
# - check whether conclusions persist when training privacy changes
#
# Sweep:
#   eps_train in {4,8}
#   eps_cal fixed at 8
#   nominal_coverage in {0.65,0.75}
#   cal_size in {2000,4000}
#   seeds in {0,1,2}
#   coverage_target fixed at 0.7
########################################
OUTDIR_C="toy_out_train_privacy_sensitivity_target070"

for seed in 0 1 2; do
  for cal_size in 2000 4000; do
    for nomcov in 0.65 0.75; do
      for teps in 4.0 8.0; do
        echo "[C] seed=$seed cal_size=$cal_size nomcov=$nomcov teps=$teps"
        python toy.py \
          --dataset "$DATASET" \
          --seed "$seed" \
          --epochs_np "$EPOCHS_NP" \
          --epochs_dp "$EPOCHS_DP" \
          --dp_eps_train "$teps" \
          --dp_eps_cal 8.0 \
          --cal_size "$cal_size" \
          --nominal_coverage "$nomcov" \
          --coverage_target 0.7 \
          --beta "$BETA" \
          --batch_size "$BATCH_SIZE" \
          --temperature "$TEMPERATURE" \
          --label_smoothing "$LABEL_SMOOTHING" \
          --verbose \
          --out_dir "$OUTDIR_C" \
          2>&1 | tee "logs/C_trainpriv_seed${seed}_cal${cal_size}_nom${nomcov}_teps${teps}.log"
      done
    done
  done
done

########################################
# Section D: High-nominal saturation check
# Purpose:
# - document saturated / pathological regime explicitly
#
# Sweep:
#   nominal_coverage in {0.85,0.95}
#   eps_cal in {2,4,8}
#   cal_size in {1000,2000}
#   seeds in {0,1,2}
#   coverage_target fixed at 0.7
########################################
OUTDIR_D="toy_out_saturation_check_target070"

for seed in 0 1 2; do
  for cal_size in 1000 2000; do
    for nomcov in 0.85 0.95; do
      for ceps in 2.0 4.0 8.0; do
        echo "[D] seed=$seed cal_size=$cal_size nomcov=$nomcov ceps=$ceps"
        python toy.py \
          --dataset "$DATASET" \
          --seed "$seed" \
          --epochs_np "$EPOCHS_NP" \
          --epochs_dp "$EPOCHS_DP" \
          --dp_eps_train "$DP_EPS_TRAIN" \
          --dp_eps_cal "$ceps" \
          --cal_size "$cal_size" \
          --nominal_coverage "$nomcov" \
          --coverage_target 0.7 \
          --beta "$BETA" \
          --batch_size "$BATCH_SIZE" \
          --temperature "$TEMPERATURE" \
          --label_smoothing "$LABEL_SMOOTHING" \
          --verbose \
          --out_dir "$OUTDIR_D" \
          2>&1 | tee "logs/D_saturation_seed${seed}_cal${cal_size}_nom${nomcov}_ceps${ceps}.log"
      done
    done
  done
done