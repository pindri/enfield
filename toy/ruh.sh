#!/bin/sh

for seed in 0 1 2 3 4; do
  for ceps in 0.1 0.25; do
    for csize in 500; do
      for nomcov in 0.909 0.910 0.911; do
        for target in 0.908 0.909 0.910; do
          python toy.py \
            --seed $seed \
            --dp_train_eps 1 \
            --dp_cal_eps $ceps \
            --cal_size $csize \
            --nominal_coverage $nomcov \
            --coverage_target $target \
            --beta 1e-3
        done
      done
    done
  done
done