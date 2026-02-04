
# -*- coding: utf-8 -*-
"""
Модели машинного и глубокого обучения.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей
в обновлении вычислительной техники.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# ------------------------------------------------------------
# Попытка импортировать PyTorch (опционально)
# ------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None


# ------------------------------------------------------------
# Классические модели sklearn
# ------------------------------------------------------------
def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=200, n_jobs=-1)


def build_knn(n_neighbors: int = 5) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")


def build_random_forest(
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )


def build_gradient_boosting() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=42)


def build_extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
    )


def build_mlp_sklearn() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=200,
        random_state=42,
    )


# ------------------------------------------------------------
# PyTorch-блок (используется только если torch установлен)
# ------------------------------------------------------------
if TORCH_AVAILABLE:

    @dataclass
    class TorchTrainingConfig:
        input_dim: int
        hidden_dim: int = 64
        lr: float = 1e-3
        batch_size: int = 32
        num_epochs: int = 20
        device: str = "cpu"

    class MLPNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def train_mlp_torch(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cfg: TorchTrainingConfig,
    ) -> Tuple[nn.Module, Dict[str, list]]:

        device = torch.device(cfg.device)
        model = MLPNet(cfg.input_dim, cfg.hidden_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

        def _make_loader(X, y):
            tensor_x = torch.tensor(X, dtype=torch.float32)
            tensor_y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
            ds = torch.utils.data.TensorDataset(tensor_x, tensor_y)
            return torch.utils.data.DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=True
            )

        train_loader = _make_loader(X_train, y_train)
        val_loader = _make_loader(X_val, y_val)

        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(cfg.num_epochs):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            preds_list = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    val_loss += loss.item() * xb.size(0)
                    preds_list.append(outputs.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            preds = np.vstack(preds_list)
            y_val_pred_labels = (preds >= 0.5).astype(int).ravel()
            val_acc = accuracy_score(y_val, y_val_pred_labels)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Эпоха {epoch + 1}/{cfg.num_epochs} - "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}"
            )

        return model, history
