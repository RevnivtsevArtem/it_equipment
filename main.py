# -*- coding: utf-8 -*-
"""
Главное web-приложение Streamlit.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
"""

from __future__ import annotations

import io
import os
import glob

import pandas as pd
import streamlit as st

# ВАЖНО: первая Streamlit-команда
st.set_page_config(
    page_title="Прогноз обновления вычислительной техники",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ----------------------- СТРАНИЦЫ -----------------------

def page_overview():
    st.subheader("Общая информация о системе")
    st.markdown("""
    Данное приложение реализует интеллектуальный сервис прогнозирования потребностей
    в обновлении вычислительной техники на основе обращений в службу технической поддержки.
    """)


def page_data(df, cfg):
    st.subheader("Работа с данными")
    st.write("Размер набора данных:", df.shape)
    st.dataframe(df.head(20), use_container_width=True)


def page_prediction(df, cfg):
    st.subheader("Прогноз необходимости замены устройства")

    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    model_paths = sorted(glob.glob("models/*.pkl"))
    if not model_paths:
        st.warning("Нет сохранённых моделей в папке models/")
        return

    model_names = [os.path.basename(p) for p in model_paths]
    selected = st.selectbox("Модель", ["ВСЕ модели"] + model_names)

    input_data = {}

    for col in cfg["categorical_columns"]:
        values = sorted(df[col].dropna().astype(str).unique())
        input_data[col] = st.selectbox(col, values)

    for col in cfg["numeric_columns"]:
        default_val = float(pd.to_numeric(df[col], errors="coerce").mean())
        input_data[col] = st.number_input(col, value=default_val)

    input_df = pd.DataFrame([input_data])

    threshold = st.slider("Порог (threshold)", 0.05, 0.95, 0.50, 0.05)

    if st.button("Спрогнозировать"):
        paths = model_paths if selected == "ВСЕ модели" else [
            p for p in model_paths if os.path.basename(p) == selected
        ]

        results = []

        for pth in paths:
            model = load_model(pth)
            _, proba = predict_needs_upgrade(
                model=model,
                df_inputs=input_df,
                feature_cols=feature_cols,
            )
            prob = float(proba[0])
            decision = "ТРЕБУЕТСЯ обновление" if prob >= threshold else "Пока не требуется"
            results.append({
                "Модель": os.path.basename(pth),
                "Вероятность": prob,
                "Решение": decision
            })

        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)


def page_eda(df):
    st.subheader("Разведочный анализ данных (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_ticket_counts_by_department(df))
    with col2:
        st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


def page_model_comparison(df, cfg):
    from sklearn.model_selection import train_test_split

    st.subheader("Сравнение моделей")

    target_col = cfg["default_target_column"]

    test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2)
    random_state = st.number_input("Random state", value=42)

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=int(random_state),
        stratify=df[target_col] if df[target_col].nunique() > 1 else None,
    )

    if st.button("Обучить и сравнить модели"):
        report_df = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=target_col,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=[
                "LogisticRegression",
                "KNN",
                "RandomForest",
                "GradientBoosting",
                "ExtraTrees",
                "MLPClassifier",
            ],
        )
        st.dataframe(report_df, use_container_width=True)


# ----------------------- MAIN -----------------------

def main():

    with st.sidebar:
        st.header("Загрузка данных")
        uploaded_file = st.file_uploader(
            "Загрузите CSV с обращениями",
            type=["csv"]
        )

        page = st.radio(
            "Раздел приложения",
            ["Обзор", "Данные", "Обучение",
             "Сравнение моделей", "Прогноз устройства",
             "EDA", "Отчёт"],
        )

    st.title("Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники")

    cfg = load_config()

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Данные загружены")
    else:
        df = _load_default_data()
        st.info("Используется демонстрационный датасет data/sample_tickets.csv")

    if page == "Обзор":
        page_overview()
    elif page == "Данные":
        page_data(df, cfg)
    elif page == "Сравнение моделей":
        page_model_comparison(df, cfg)
    elif page == "Прогноз устройства":
        page_prediction(df, cfg)
    elif page == "EDA":
        page_eda(df)


if __name__ == "__main__":
    main()
