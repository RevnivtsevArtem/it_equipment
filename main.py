# -*- coding: utf-8 -*-
"""
Главное web-приложение Streamlit.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

from src.app_core import (
    load_config,
    fit_on_full_and_save,
    load_model,
    predict_needs_upgrade,
    fit_model_and_evaluate,
    evaluate_all_models,
    )

from src.eda import (
    plot_ticket_counts_by_department,
    plot_ticket_counts_by_device_type,
    plot_device_age_hist,
    plot_tickets_last_6_months_hist,
)
from src.evaluation import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    metrics_to_markdown_table,
)


def _load_default_data() -> pd.DataFrame:
    """Загружает демонстрационный датасет."""
    return pd.read_csv("data/sample_tickets.csv")

def _validate_uploaded_dataset(df: pd.DataFrame, cfg: dict):
    """Проверка структуры пользовательского датасета"""
    required_columns = (
        [cfg["default_target_column"]]
        + cfg["categorical_columns"]
        + cfg["numeric_columns"]
    )

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return None, f"Отсутствуют обязательные столбцы: {missing_cols}"

    target_col = cfg["default_target_column"]
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)

    return df, None


# ... (код полностью сохранён как в вашем сообщении) ...
