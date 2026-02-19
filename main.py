# -*- coding: utf-8 -*-
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
    return pd.read_csv("data/sample_tickets.csv")


def page_overview() -> None:
    st.subheader("Общая информация о системе")
    st.markdown(
        "Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники."
    )


def page_data(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Работа с данными")
    st.write("Размер набора данных:", df.shape)
    st.dataframe(df.head(20))


def page_training(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Обучение и сохранение модели")
    target_col = cfg["default_target_column"]

    model_name = st.selectbox(
        "Выберите модель для обучения",
        ["LogisticRegression", "KNN", "RandomForest", "GradientBoosting", "ExtraTrees", "MLPClassifier"],
    )

    model_filename = f"{model_name.lower()}_full.pkl"
    model_path = f"models/{model_filename}"

    if st.button("Обучить модель на всём датасете и сохранить"):
        pipeline, metrics = fit_on_full_and_save(
            df=df,
            target_column=target_col,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_name=model_name,
            model_path=model_path,
        )
        st.success(f"Модель сохранена: {model_path}")
        st.json(metrics)


def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    from sklearn.model_selection import train_test_split

    st.subheader("Сравнение моделей")
    target_col = cfg["default_target_column"]
    df_split = df.dropna(subset=[target_col]).copy()

    df_train, df_test = train_test_split(df_split, test_size=0.2, random_state=42)

    all_models = [
        "LogisticRegression",
        "KNN",
        "RandomForest",
        "GradientBoosting",
        "ExtraTrees",
        "MLPClassifier",
    ]

    if st.button("Обучить и сравнить модели"):
        report_df = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=target_col,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=all_models,
        )
        st.dataframe(report_df)


def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз необходимости замены устройства")

    models_dir = "models"
    if not os.path.exists(models_dir):
        st.warning("Папка models отсутствует.")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]

    if not model_files:
        st.warning("Нет сохранённых моделей.")
        return

    selected_model_file = st.selectbox("Выберите обученную модель", model_files)
    model_path = os.path.join(models_dir, selected_model_file)
    model = load_model(model_path)

    input_data = {}
    for col in cfg["categorical_columns"]:
        input_data[col] = st.selectbox(col, sorted(df[col].dropna().unique()))

    for col in cfg["numeric_columns"]:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("Рассчитать вероятность"):
        input_df = pd.DataFrame([input_data])
        feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]
        labels, probabilities = predict_needs_upgrade(
            model=model,
            df_inputs=input_df,
            feature_cols=feature_cols,
        )
        proba_percent = float(probabilities[0]) * 100
        st.progress(int(proba_percent))
        st.write(f"Вероятность замены: {proba_percent:.2f}%")


def main() -> None:
    st.set_page_config(page_title="Прогноз обновления вычислительной техники", layout="wide")
    st.title("Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники")

    cfg = load_config()

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = _load_default_data()

    page = st.sidebar.radio(
        "Раздел приложения",
        ["Обзор", "Данные", "Обучение", "Сравнение моделей", "Прогноз"],
    )

    if page == "Обзор":
        page_overview()
    elif page == "Данные":
        page_data(df, cfg)
    elif page == "Обучение":
        page_training(df, cfg)
    elif page == "Сравнение моделей":
        page_model_comparison(df, cfg)
    elif page == "Прогноз":
        page_prediction(df, cfg)


if __name__ == "__main__":
    main()
