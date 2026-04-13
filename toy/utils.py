import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from conformal import aps_prediction_set_mask


# -----------------------------
# Random utils.
# -----------------------------

def make_reproducible(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_of_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Models.
# -----------------------------

class TinyMLP(nn.Module):
    def __init__(self, in_dim=28 * 28, hidden=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TinyCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MediumPowerfulCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def inject_label_noise(subset, noise_rate: float, num_classes: int, seed: int):
    if noise_rate <= 0.0:
        return

    g = torch.Generator().manual_seed(seed)
    base_ds = subset.dataset
    idxs = subset.indices
    n_flip = int(len(idxs) * noise_rate)
    perm = torch.randperm(len(idxs), generator=g)[:n_flip]

    if hasattr(base_ds, "targets"):
        for p in perm.tolist():
            j = idxs[p]
            old = int(base_ds.targets[j])
            new = torch.randint(low=0, high=num_classes - 1, size=(1,), generator=g).item()
            if new >= old:
                new += 1
            base_ds.targets[j] = new
    else:
        raise ValueError("Dataset does not expose .targets for label corruption")

# -----------------------------
# Conformal stuff.
# -----------------------------

@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: str, temperature: float = 1.0):
    model.eval()
    probs_list = []
    labels_list = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = F.softmax(logits / temperature, dim=1)
        probs_list.append(probs.cpu())
        labels_list.append(y.cpu())
    return torch.cat(probs_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def evaluate_coverage_aps(model: nn.Module, loader: DataLoader, tau: float, device: str, temperature: float = 1.0) -> Dict[str, float]:
    probs, labels = predict_proba(model, loader, device, temperature=temperature)
    N, K = probs.shape
    covered = 0
    total_set_size = 0.0
    min_set = 10**9

    for i in range(N):
        mask = aps_prediction_set_mask(probs[i], tau)
        sz = mask.sum().item()
        total_set_size += sz
        min_set = min(min_set, sz)
        if mask[labels[i].item()].item():
            covered += 1

    return {
        "coverage": covered / N,
        "avg_set_size": total_set_size / N,
        "min_set_size": float(min_set),
    }


# -----------------------------
# Training.
# -----------------------------

def train_epoch(model, loader, optimizer, device, label_smoothing=0.0) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if label_smoothing > 0.0:
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        else:
            loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct / max(total, 1)


# -----------------------------
# Reporting and feasibility.
# -----------------------------

@dataclass
class Contract:
    epsilon_train: float
    delta: float
    epsilon_cal: float
    coverage_target: float
    alpha: float
    num_bins: int

    def is_feasible(self,) -> bool:
        return True


def load_reports(report_dir: str | Path) -> pd.DataFrame:
    """
    Load all report_*.json files and flatten the main fields used for plotting.
    """
    report_dir = Path(report_dir)
    rows = []

    for path in sorted(report_dir.glob("report_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)

        row = {
            "file": path.name,

            # Contract.
            "alpha": r["contract"]["alpha"],
            "coverage_target": r["contract"]["coverage_target"],
            "delta": r["contract"]["delta"],
            "epsilon_train_target": r["contract"]["epsilon_train"],
            "epsilon_cal": r["contract"]["epsilon_cal"],
            "num_bins": r["contract"]["num_bins"],

            # Meta.
            "cal_size": r["meta"]["cal_size"],
            "train_size": r["meta"]["train_size"],
            "seed": r["meta"]["seed"],
            "device": r["meta"]["device"],
            "nominal_coverage": r["meta"]["nominal_coverage"],
            "label_smoothing": r["meta"]["label_smoothing"],

            # Non-private baseline.
            "np_accuracy": r["non_private"]["test_accuracy"],
            "np_coverage": r["non_private"]["test_coverage"],
            "np_avg_set_size": r["non_private"]["avg_set_size"],
            "np_tau": r["non_private"]["tau"],

            # DP training.
            "dp_accuracy": r["dp_training"]["test_accuracy"],
            "epsilon_train_realized": r["dp_training"]["epsilon_realized"],
            "noise_multiplier": r["dp_training"]["noise_multiplier"],
            "tau_nonprivate_cal": r["dp_training"]["tau_nonprivate_cal"],
            "test_coverage_nonprivate_cal": r["dp_training"]["test_coverage_nonprivate_cal"],
            "dp_avg_set_size_nonprivate_cal": r["dp_training"]["avg_set_size_nonprivate_cal"],

            # DP composition.
            "epsilon_total_basic_composition": r["privacy_composition"]["epsilon_total_basic_composition"],

            # DP calibration.
            "beta": r["dp_calibration"]["beta"],
            "coverage_lower_bound_formal": r["dp_calibration"]["coverage_lower_bound_formal"],
            "tau_dp_cal": r["dp_calibration"]["tau_dp_cal"],
            "test_coverage_dp_cal": r["dp_calibration"]["test_coverage_dp_cal"],
            "avg_set_size_dp_cal": r["dp_calibration"]["avg_set_size_dp_cal"],
            "lambda": r["dp_calibration"]["lambda"],
            "noise_scale": r["dp_calibration"]["noise_scale"],
            "k": r["dp_calibration"]["k"],

            # Pass/Fail.
            "coverage_bound_formal_ok": r["pass_fail"]["coverage_bound_formal_ok"],
            "privacy_training_ok": r["pass_fail"]["privacy_training_ok"],
            "coverage_empirical_ok": r["pass_fail"]["coverage_empirical_ok"],
            "overall_formal_ok": r["pass_fail"]["overall_formal_ok"],
            "overall_empirical_ok": r["pass_fail"]["overall_empirical_ok"],

            # Additional threshold info.
            "tau_q_k": r["dp_calibration"]["tau_q_k"],
            "tau_q_kplus2lambda": r["dp_calibration"]["tau_q_kplus2lambda"],
            "certificate_width_tau": r["dp_calibration"]["certificate_width_tau"],
            "observed_inflation_tau_grid": r["dp_calibration"]["observed_inflation_tau_grid"],
            "observed_inflation_tau_exact": r["dp_calibration"]["observed_inflation_tau_exact"],
            "theorem_tau_ok": r["dp_calibration"]["theorem_tau_ok"],
            "theorem_idx_ok": r["dp_calibration"]["theorem_idx_ok"],
            "q_k_index": r["dp_calibration"]["q_k_index"],
            "q_kplus2lambda_index": r["dp_calibration"]["q_kplus2lambda_index"],
            "crossing_index_dp": r["dp_calibration"]["crossing_index_dp"],
        }

        # Useful derived fields.
        row["empirical_margin_to_bound"] = (
            row["test_coverage_dp_cal"] - row["coverage_lower_bound_formal"]
        )
        row["empirical_margin_to_target"] = (
            row["test_coverage_dp_cal"] - row["coverage_target"]
        )
        row["empirical_margin_to_nominal"] = (
            row["test_coverage_dp_cal"] - row["nominal_coverage"]
        )
        row["inflation_to_certificate_ratio"] = (
            row["observed_inflation_tau_grid"] / row["certificate_width_tau"]
            if row["certificate_width_tau"] > 0 else np.nan
        )

        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No report_*.json files found in {report_dir}")

    df = pd.DataFrame(rows)
    return df


def save_dataframe(df: pd.DataFrame, out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
