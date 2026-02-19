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
    return pd.read_csv("data/sample_tickets.csv")


def page_overview() -> None:
    st.subheader("Общая информация о системе")
    st.markdown(
        """
        Интеллектуальная система прогнозирования потребностей
        в обновлении вычислительной техники.
        """
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
        ["LogisticRegression", "KNN", "RandomForest",
         "GradientBoosting", "ExtraTrees", "MLPClassifier"],
    )

    model_path = f"models/{model_name.lower()}_full.pkl"

    if st.button("Обучить модель на всём датасете и сохранить"):
        with st.spinner("Идёт обучение модели..."):
            fit_on_full_and_save(
                df=df,
                target_column=target_col,
                categorical_cols=cfg["categorical_columns"],
                numeric_cols=cfg["numeric_columns"],
                model_name=model_name,
                model_path=model_path,
            )
        st.success(f"Модель сохранена в {model_path}")


# ===================== ИСПРАВЛЕННАЯ ФУНКЦИЯ =====================

def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз необходимости замены устройства")

    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    if df_clean.empty:
        st.error("Нет корректных данных для прогнозирования.")
        return

    selected_index = st.selectbox(
        "Выберите устройство (строку датасета)",
        df_clean.index,
    )

    st.write("Параметры выбранного устройства:")
    st.dataframe(df_clean.loc[[selected_index]])

    if st.button("Спрогнозировать по всем моделям"):

        df_input = df_clean.loc[[selected_index]]

        model_names = [
            "LogisticRegression",
            "KNN",
            "RandomForest",
            "GradientBoosting",
            "ExtraTrees",
            "MLPClassifier",
        ]

        results = []

        for model_name in model_names:
            model_path = f"models/{model_name.lower()}_full.pkl"

            if not os.path.exists(model_path):
                continue

            model = load_model(model_path)

            _, proba = predict_needs_upgrade(
                model=model,
                df_inputs=df_input,
                feature_cols=feature_cols,
            )

            results.append({
                "model": model_name,
                "probability": float(proba[0]),
            })

        if not results:
            st.error("Нет сохранённых моделей.")
            return

        results_df = pd.DataFrame(results)

        st.subheader("Вероятности по каждой модели")
        st.dataframe(results_df)

        avg_proba = results_df["probability"].mean()

        st.subheader("Ансамблевый прогноз (среднее)")
        st.progress(avg_proba)
        st.markdown(f"### Средняя вероятность: {avg_proba:.2%}")

# ================================================================


def main() -> None:
    st.set_page_config(
        page_title="Прогноз обновления вычислительной техники",
        layout="wide"
    )

    st.title("Интеллектуальная система прогнозирования потребностей")

    cfg = load_config()

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = _load_default_data()

    page = st.sidebar.radio(
        "Раздел приложения",
        ["Обзор", "Данные", "Обучение", "Прогноз устройства"],
    )

    if page == "Обзор":
        page_overview()
    elif page == "Данные":
        page_data(df, cfg)
    elif page == "Обучение":
        page_training(df, cfg)
    elif page == "Прогноз устройства":
        page_prediction(df, cfg)


if __name__ == "__main__":
    main()
