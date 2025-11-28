# -*- coding: utf-8 -*-
"""
Модуль предобработки данных.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> ColumnTransformer:
    """Формирует конвейер предобработки признаков."""
    categorical_transformer = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features_target(
    df: pd.DataFrame,
    target_column: str,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[str]]]:
    """Делит датафрейм на матрицу признаков X и вектор целей y."""
    missing_cols = [c for c in categorical_cols + numeric_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"В датасете отсутствуют необходимые столбцы: {missing_cols}")

    if target_column not in df.columns:
        raise ValueError(f"Целевой столбец '{target_column}' не найден.")

    df = df.copy()
    df = df.dropna(subset=[target_column])

    X = df[categorical_cols + numeric_cols]
    y = df[target_column].astype(int).values

    meta = {
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }
    return X, y, meta


def train_test_split_stratified(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Делит выборку на обучающую и тестовую с сохранением пропорций классов."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
