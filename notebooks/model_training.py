# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from src.data_cleaner import clean_dataset
from src.models import (
    build_logistic_regression,
    build_knn,
    build_random_forest,
    build_gradient_boosting,
    build_extra_trees,
    build_mlp_sklearn,
)
from src.evaluation import evaluate_binary_classifier
from sklearn.model_selection import train_test_split

DATA_PATH = "data/sample_tickets.csv"
TARGET_COLUMN = "needs_upgrade"

CATEGORICAL_COLS = [
    "user_department",
    "device_type",
    "priority",
    "os",
    "location",
]

NUMERIC_COLS = [
    "device_age_years",
    "tickets_last_6_months",
]

def run_training():
    df = pd.read_csv(DATA_PATH)
    df = clean_dataset(df)

    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN],
    )

    models = {
        "LogisticRegression": build_logistic_regression,
        "KNN": build_knn,
        "RandomForest": build_random_forest,
        "GradientBoosting": build_gradient_boosting,
        "ExtraTrees": build_extra_trees,
        "MLPClassifier": build_mlp_sklearn,
    }

    results = []

    for model_name, builder in models.items():
        model = builder(
            categorical_cols=CATEGORICAL_COLS,
            numeric_cols=NUMERIC_COLS,
        )

        X_train = df_train[CATEGORICAL_COLS + NUMERIC_COLS]
        y_train = df_train[TARGET_COLUMN]
        X_test = df_test[CATEGORICAL_COLS + NUMERIC_COLS]
        y_test = df_test[TARGET_COLUMN]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_binary_classifier(y_test, y_pred, y_proba)
        metrics["model"] = model_name
        results.append(metrics)

    report_df = pd.DataFrame(results)
    print(report_df)

if __name__ == "__main__":
    run_training()
