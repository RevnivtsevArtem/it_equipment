# -*- coding: utf-8 -*-
"""
Оценка качества моделей и визуализация.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
"""

from __future__ import annotations

from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred_labels: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Вычисляет набор стандартных метрик для задач бинарной классификации."""
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred_labels)),
        "precision": float(precision_score(y_true, y_pred_labels, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_labels, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_labels, zero_division=0)),
    }
    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred_labels: np.ndarray) -> plt.Figure:
    """Строит матрицу ошибок для бинарной классификации."""
    cm = confusion_matrix(y_true, y_pred_labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> plt.Figure:
    """Строит ROC-кривую."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pr_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> plt.Figure:
    """Строит PR-кривую (precision-recall)."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall кривая")
    ax.legend()
    fig.tight_layout()
    return fig


def metrics_to_markdown_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """Преобразует словарь метрик по моделям в markdown-таблицу."""
    headers = ["Модель", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for model_name, vals in metrics.items():
        row = [
            model_name,
            f"{vals.get('accuracy', float('nan')):.3f}",
            f"{vals.get('precision', float('nan')):.3f}",
            f"{vals.get('recall', float('nan')):.3f}",
            f"{vals.get('f1', float('nan')):.3f}",
            f"{(vals.get('roc_auc') or float('nan')):.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
