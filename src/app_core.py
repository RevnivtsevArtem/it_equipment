# -*- coding: utf-8 -*-
"""
Вспомогательные функции для работы web-приложения.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessing_pipeline
from .models import (
    build_logistic_regression,
    build_knn,
    build_random_forest,
    build_gradient_boosting,
    build_extra_trees,
    build_mlp_sklearn,
)
from .evaluation import evaluate_binary_classifier


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Загружает JSON-конфигурацию проекта."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_by_name(name: str):
    """Возвращает экземпляр модели по её названию."""
    if name == "LogisticRegression":
        return build_logistic_regression()
    if name == "KNN":
        return build_knn()
    if name == "RandomForest":
        return build_random_forest()
    if name == "GradientBoosting":
        return build_gradient_boosting()
    if name == "ExtraTrees":
        return build_extra_trees()
    if name == "MLPClassifier":
        return build_mlp_sklearn()
    raise ValueError(f"Неизвестная модель: {name}")


def build_pipeline(
    model_name: str,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> Pipeline:
    """Формирует sklearn-пайплайн 'предобработка + модель'."""
    preprocessor = build_preprocessing_pipeline(categorical_cols, numeric_cols)
    model = build_model_by_name(model_name)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def fit_model_and_evaluate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_column: str,
    categorical_cols: List[str],
    numeric_cols: List[str],
    model_name: str,
) -> Tuple[Pipeline, Dict[str, float]]:
    """Обучает модель на обучающей выборке и оценивает её на тестовой."""
    X_train = df_train[categorical_cols + numeric_cols]
    y_train = df_train[target_column].astype(int).values

    X_test = df_test[categorical_cols + numeric_cols]
    y_test = df_test[target_column].astype(int).values

    pipeline = build_pipeline(model_name, categorical_cols, numeric_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = evaluate_binary_classifier(y_test, y_pred, y_proba)
    return pipeline, metrics


def fit_on_full_and_save(
    df: pd.DataFrame,
    target_column: str,
    categorical_cols: List[str],
    numeric_cols: List[str],
    model_name: str,
    model_path: str,
) -> Tuple[Pipeline, Dict[str, float]]:
    """Обучает модель на полном датасете и сохраняет её в файл."""
    X = df[categorical_cols + numeric_cols]
    y = df[target_column].astype(int).values

    pipeline = build_pipeline(model_name, categorical_cols, numeric_cols)
    pipeline.fit(X, y)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X)[:, 1]
    else:
        y_proba = None
    y_pred = pipeline.predict(X)
    metrics = evaluate_binary_classifier(y, y_pred, y_proba)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    return pipeline, metrics


def load_model(model_path: str) -> Pipeline:
    """Загружает сохранённую модель."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели {model_path} не найден.")
    model: Pipeline = joblib.load(model_path)
    return model


def predict_needs_upgrade(
    model: Pipeline,
    df_inputs: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Применяет модель к набору устройств и возвращает метки и вероятности."""
    X = df_inputs[feature_cols]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)
        proba = preds.astype(float)
    labels = (proba >= 0.5).astype(int)
    return labels, proba
