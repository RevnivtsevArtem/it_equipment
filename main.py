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


def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


def _validate_uploaded_dataset(df: pd.DataFrame, cfg: dict):
    required_columns = (
        [cfg["default_target_column"]]
        + cfg["categorical_columns"]
        + cfg["numeric_columns"]
    )

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        return None, f"Отсутствуют обязательные столбцы: {missing}"

    target = cfg["default_target_column"]
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    return df, None


def page_data(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Работа с данными")
    st.write("Размер набора данных:", df.shape)
    st.dataframe(df.head(20))

    if st.checkbox("Показать статистику числовых признаков"):
        st.write(df[cfg["numeric_columns"]].describe())

    if st.checkbox("Скачать текущий датасет в CSV"):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "Скачать CSV",
            data=buf.getvalue(),
            file_name="current_dataset.csv",
            mime="text/csv",
        )


def page_training(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Обучение и сохранение модели")

    target_col = cfg["default_target_column"]

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует.")
        return

    model_name = st.selectbox(
        "Выберите модель",
        ["LogisticRegression", "KNN", "RandomForest", "GradientBoosting", "ExtraTrees", "MLPClassifier"],
    )

    model_filename = st.text_input("Имя файла модели", value=f"{model_name.lower()}_full.pkl")
    model_path = f"models/{model_filename}"

    if st.button("Обучить модель"):
        with st.spinner("Обучение..."):
            _, metrics = fit_on_full_and_save(
                df=df,
                target_column=target_col,
                categorical_cols=cfg["categorical_columns"],
                numeric_cols=cfg["numeric_columns"],
                model_name=model_name,
                model_path=model_path,
            )
        st.success(f"Модель сохранена: {model_path}")
        st.json(metrics)


def main() -> None:
    st.set_page_config(page_title="Прогноз обновления вычислительной техники", layout="wide")
    st.title("Интеллектуальная система прогнозирования обновления техники")

    cfg = load_config()

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            df_validated, error = _validate_uploaded_dataset(df_uploaded, cfg)

            if error:
                st.sidebar.error(error)
                df = _load_default_data()
                st.sidebar.info("Используется демонстрационный датасет.")
            else:
                df = df_validated
                st.sidebar.success("Пользовательский датасет загружен.")
        except Exception as e:
            st.sidebar.error(f"Ошибка: {e}")
            df = _load_default_data()
    else:
        df = _load_default_data()
        st.sidebar.info("Используется демонстрационный датасет.")

    page = st.sidebar.radio(
        "Раздел",
        ["Данные", "Обучение"],
    )

    if page == "Данные":
        page_data(df, cfg)
    elif page == "Обучение":
        page_training(df, cfg)


if __name__ == "__main__":
    main()
