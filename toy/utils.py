import os
import random
from dataclasses import dataclass
from typing import Dict

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
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
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
