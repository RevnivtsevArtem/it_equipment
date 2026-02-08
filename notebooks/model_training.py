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

DATA_PATH = "data/sample_tickets.csv"
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

df = pd.read_csv(DATA_PATH)
df = clean_dataset(df)

X = df[categorical_cols + numeric_cols]
y = df[TARGET_COL].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
    ]
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=300, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
}

results = {}

for name, model in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"Модель {name}: accuracy = {acc:.3f}")

best_model = max(results, key=results.get)
print(f"\nЛучшая модель: {best_model} (accuracy = {results[best_model]:.3f})")
