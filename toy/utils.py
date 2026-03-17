import os
import random
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from conformal import aps_prediction_set_mask, prediction_set_from_probs


# -----------------------------
# Random utils.
# -----------------------------
def set_seed(seed: int) -> None:
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


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper to predict probabilities and labels.

    Returns:
      probs: (N, K)
      labels: (N,)
    """
    model.eval()
    probs_list = []
    labels_list = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probs_list.append(probs.cpu())
        labels_list.append(y.cpu())
    return torch.cat(probs_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def evaluate_coverage(model: nn.Module, loader: DataLoader, tau: float, device: str) -> Dict[str, float]:
    """
    Evaluate achieved coverage and average set size on a loader.
    """
    probs, labels = predict_proba(model, loader, device)
    N, K = probs.shape
    covered = 0
    total_set_size = 0.0
    for i in range(N):
        mask = prediction_set_from_probs(probs[i], tau)
        total_set_size += mask.sum().item()
        if mask[labels[i].item()].item():
            covered += 1
    return {
        "coverage": covered / N,
        "avg_set_size": total_set_size / N,
    }



@torch.no_grad()
def evaluate_coverage_aps(model: nn.Module, loader: DataLoader, tau: float, device: str) -> Dict[str, float]:
    """
    Evaluate achieved coverage and average APS set size.
    """
    probs, labels = predict_proba(model, loader, device)
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
def train_epoch(model, loader, optimizer, device, dp: bool = False) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
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
# Reporting
# -----------------------------
@dataclass
class Contract:
    epsilon_train: float
    delta: float
    epsilon_cal: float
    coverage_target: float
    alpha: float
    num_bins: int


