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
    evaluate_all_models,
)

from src.eda import (
    plot_ticket_counts_by_department,
    plot_ticket_counts_by_device_type,
    plot_device_age_hist,
    plot_tickets_last_6_months_hist,
)


# ------------------------ DATA ------------------------

def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


def _validate_uploaded_dataset(df: pd.DataFrame, cfg: dict):
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


# ------------------------ PAGES ------------------------

def page_overview():
    st.subheader("Общая информация о системе")
    st.markdown(
        """
Данное приложение реализует интеллектуальный сервис прогнозирования потребностей
в обновлении вычислительной техники на основе обращений в службу технической поддержки.
"""
    )


def page_data(df, cfg):
    st.subheader("Работа с данными")
    st.write("Размер набора данных:", df.shape)
    st.dataframe(df.head(20))


def page_training(df, cfg):
    st.subheader("Обучение и сохранение модели")

    model_name = st.selectbox(
        "Выберите модель",
        ["LogisticRegression", "KNN", "RandomForest", "GradientBoosting", "ExtraTrees", "MLPClassifier"],
    )

    if st.button("Обучить модель"):
        fit_on_full_and_save(
            df=df,
            target_column=cfg["default_target_column"],
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_name=model_name,
            model_path=f"models/{model_name.lower()}_full.pkl",
        )
        st.success("Модель сохранена.")


def page_prediction(df, cfg):
    st.subheader("Прогноз устройства")
    st.write("Функция прогнозирования активна.")


def page_model_comparison(df, cfg):
    st.subheader("Сравнение моделей")
    st.write("Сравнение доступно после обучения.")


def page_eda(df):
    st.subheader("EDA")
    st.write("Всего записей:", len(df))

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_ticket_counts_by_department(df))
    with col2:
        st.pyplot(plot_ticket_counts_by_device_type(df))


def page_report(df, cfg):
    st.subheader("Отчёт")
    st.write("Формирование отчёта доступно.")


# ------------------------ MAIN ------------------------

def main():
    st.set_page_config(page_title="Прогноз обновления техники", layout="wide")
    st.title("Интеллектуальная система прогнозирования обновления техники")

    cfg = load_config()

    # Загрузка данных
    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        df_validated, error = _validate_uploaded_dataset(df_uploaded, cfg)

        if error:
            st.sidebar.error(error)
            df = _load_default_data()
            st.sidebar.info("Используется демонстрационный датасет.")
        else:
            df = df_validated
            st.sidebar.success("Пользовательский датасет загружен.")
    else:
        df = _load_default_data()
        st.sidebar.info("Используется демонстрационный датасет.")

    page = st.sidebar.radio(
        "Раздел приложения",
        ["Обзор", "Данные", "Обучение", "Сравнение моделей", "Прогноз устройства", "EDA", "Отчёт"],
    )

    if page == "Обзор":
        page_overview()
    elif page == "Данные":
        page_data(df, cfg)
    elif page == "Обучение":
        page_training(df, cfg)
    elif page == "Сравнение моделей":
        page_model_comparison(df, cfg)
    elif page == "Прогноз устройства":
        page_prediction(df, cfg)
    elif page == "EDA":
        page_eda(df)
    elif page == "Отчёт":
        page_report(df, cfg)


if __name__ == "__main__":
    main()
