# -*- coding: utf-8 -*-
"""
Обучение и сравнение моделей машинного обучения.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей
в обновлении вычислительной техники.
"""

from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier

from src.data_cleaner import clean_dataset


DATA_PATH = "../data/sample_tickets.csv"
TARGET_COL = "needs_upgrade"

categorical_cols = [
    "user_department",
    "device_type",
    "priority",
    "os",
    "location",
]

numeric_cols = [
    "device_age_years",
    "tickets_last_6_months",
]


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_dataset(df)
    return df


def build_logistic_regression():
    return LogisticRegression(max_iter=200, n_jobs=-1)


def build_knn():
    return KNeighborsClassifier(n_neighbors=5, weights="distance")


def build_random_forest():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def build_gradient_boosting():
    return GradientBoostingClassifier(random_state=42)


def build_extra_trees():
    return ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1)


def build_mlp_sklearn():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=200,
        random_state=42,
    )


if __name__ == "__main__":
    df = load_and_clean_data(DATA_PATH)

    X = df[categorical_cols + numeric_cols]
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )

    models = {
        "LogisticRegression": build_logistic_regression(),
        "KNN": build_knn(),
        "RandomForest": build_random_forest(),
        "GradientBoosting": build_gradient_boosting(),
        "ExtraTrees": build_extra_trees(),
        "MLPClassifier": build_mlp_sklearn(),
    }

    print("Результаты обучения моделей:\n")

    results = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"Модель {name}: accuracy = {acc:.3f}")

    best_model = max(results, key=results.get)
    print("\nЛучшая модель:", best_model, "с accuracy =", results[best_model])
