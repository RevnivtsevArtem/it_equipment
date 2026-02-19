
# -*- coding: utf-8 -*-
from __future__ import annotations

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

from src.evaluation import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    metrics_to_markdown_table,
)


def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


# -------------------- PAGES --------------------

def page_overview():
    st.subheader("Общая информация о системе")


def page_data(df, cfg):
    st.subheader("Работа с данными")
    st.write(f"Размер набора данных: {df.shape}")
    st.dataframe(df.head())


def page_training(df, cfg):
    st.subheader("Обучение моделей")


def page_model_comparison(df, cfg):
    st.subheader("Сравнение моделей")


def page_prediction(df, cfg):
    st.subheader("Прогноз устройства")


def page_eda(df):
    st.subheader("EDA")


def page_report(df, cfg):
    st.subheader("Отчёт")


# -------------------- MAIN --------------------

def main():

    st.set_page_config(layout="wide")
    st.title("Интеллектуальная система прогнозирования потребностей")

    cfg = load_config()

    # ---------- Correct Sidebar Loader ----------
    with st.sidebar:

        st.header("Загрузка данных")

        uploaded_file = st.file_uploader(
            "Загрузите CSV с обращениями",
            type=["csv"],
            key="file_uploader"
        )

        if "user_df" not in st.session_state:
            st.session_state.user_df = None

        if uploaded_file is not None:
            st.session_state.user_df = pd.read_csv(uploaded_file)
            st.success("Пользовательский датасет загружен")

        if st.button("Сбросить пользовательский датасет"):
            st.session_state.user_df = None
            st.rerun()

    if st.session_state.get("user_df") is not None:
        df = st.session_state.user_df
    else:
        df = _load_default_data()
        st.info("Используется демонстрационный датасет data/sample_tickets.csv")

    # ---------- Menu ----------
    page = st.sidebar.radio(
        "Раздел приложения",
        [
            "Обзор",
            "Данные",
            "Обучение",
            "Сравнение моделей",
            "Прогноз устройства",
            "EDA",
            "Отчёт",
        ],
    )

    # ---------- Routing ----------
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
