import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLP(nn.Module):
    def __init__(self, in_dim=28 * 28, hidden=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TinyCNNFMNIST(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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


def inject_label_noise(subset, noise_rate: float, num_classes: int, seed: int) -> None:
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
