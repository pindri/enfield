# Provable Policy-Driven Privacy Compliance

Differentially private conformal prediction experiments for auditability and compliance with privacy contracts.

## Environment

```bash
conda env create -f environment.yml
conda activate comply
```

All commands below should be run from `src/`.

## Running experiments

Each script runs a sweep of `toy.py` and then calls `collect_reports.py` to aggregate 
the JSON outputs into a CSV under `analysis/`.

| Script | What it covers |
|---|---|
| `run_main_mechanism.sh` | CIFAR10 main sweep (vary ε_cal, nominal coverage, cal size) |
| `run_requirements.sh` | CIFAR10 feasibility heatmaps for targets 0.6 / 0.7 / 0.8 |
| `run_train_privacy.sh` | CIFAR10 training-privacy sensitivity (vary ε_train) |
| `run_saturation.sh` | CIFAR10 saturation check in high-nominal regime |
| `run_fmnist_main_mechanism.sh` | FashionMNIST main sweep |
| `cifar100_test.sh` | CIFAR100 diagnostic |
| `fmnist_test.sh` | FashionMNIST diagnostic |

```bash
bash script_name.sh
```

To re-aggregate results from an existing `toy_out_*` directory without re-running:

```bash
python collect_reports.py --out_dir toy_out_<name> --analysis_out_dir analysis/<name>
```

## Plotting

After running the experiments, generate the plots with:

```bash
python make_plots.py
python plot_three_panels.py \
  --csv analysis/fmnist_main_mechanism_target070/all_reports.csv \
  --out analysis/fmnist_main_mechanism_target070/three_panel_main_results.pdf
python plot_three_panels.py \
  --csv analysis/main_mechanism_target070/all_reports.csv \
  --out analysis/main/three_panel_main_results.pdf
```

## Contract compiler

To search for a configuration satisfying a coverage and privacy contract:

```bash
python compiler.py
```

Edit the `grid` and `contract` in `compiler.py:main()` to change the search space and requirements.
Results are saved to `contracts/`.

## Quick test

```bash
bash test_run.sh
```
