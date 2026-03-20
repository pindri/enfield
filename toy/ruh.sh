#!/bin/sh

# Grey-zone regions with little slack.
#for seed in 0 1 2 3 4; do
#  for ceps in 0.1 0.25; do
#    for csize in 500; do
#      for nomcov in 0.909 0.910 0.911; do
#        for target in 0.908 0.909 0.910; do
#          python toy.py \
#            --seed $seed \
#            --dp_train_eps 1 \
#            --dp_cal_eps $ceps \
#            --cal_size $csize \
#            --nominal_coverage $nomcov \
#            --coverage_target $target \
#            --beta 1e-3
#        done
#      done
#    done
#  done
#done

#for seed in 0 1 2; do
#  for ceps in 0.1 1.0; do
#    for csize in 500; do
#      for nomcov in 0.90 0.92 0.94 0.96; do
#        for target in 0.89 0.90 0.91 0.92 0.93 0.94 0.95; do
#          python toy.py \
#            --seed $seed \
#            --dp_train_eps 1 \
#            --dp_cal_eps $ceps \
#            --cal_size $csize \
#            --nominal_coverage $nomcov \
#            --coverage_target $target \
#            --beta 1e-3
#        done
#      done
#    done
#  done
#done


python toy.py \
  --dataset cifar10 \
  --seed 0 \
  --epochs_np 10 \
  --epochs_dp 10 \
  --dp_train_eps 1.0 \
  --dp_cal_eps 1.0 \
  --cal_size 2000 \
  --nominal_coverage 0.7 \
  --coverage_target 0.7 \
  --beta 1e-3 \
  --batch_size 128 \
  --temperature 2.0 \
  --verbose
