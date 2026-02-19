# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os
import glob
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


# =================================================
# DEMO DATA
# =================================================

def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


# =================================================
# OVERVIEW
# =================================================

def page_overview():
    st.subheader("Общая информация")
    st.markdown(
        """
        Система прогнозирования необходимости обновления вычислительной техники
        на основе анализа обращений в службу технической поддержки.
        """
    )


# =================================================
# DATA
# =================================================

def page_data(df: pd.DataFrame, cfg: dict):
    st.subheader("Работа с данными")

    st.write("Размер датасета:", df.shape)
    st.dataframe(df.head(20), use_container_width=True)

    if st.checkbox("Показать статистику числовых признаков"):
        st.write(df[cfg["numeric_columns"]].describe())

    if st.checkbox("Скачать текущий датасет"):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "Скачать CSV",
            data=buf.getvalue(),
            file_name="dataset.csv",
            mime="text/csv"
        )


# =================================================
# TRAINING
# =================================================

def page_training(df: pd.DataFrame, cfg: dict):

    st.subheader("Обучение модели")

    target_col = cfg["default_target_column"]

    if target_col not in df.columns:
        st.error("Целевой столбец отсутствует.")
        return

    model_name = st.selectbox(
        "Модель",
        ["LogisticRegression", "KNN", "RandomForest",
         "GradientBoosting", "ExtraTrees", "MLPClassifier"]
    )

    model_filename = st.text_input("Имя файла модели", f"{model_name.lower()}_full.pkl")
    model_path = f"models/{model_filename}"

    if st.button("Обучить и сохранить"):
        with st.spinner("Обучение..."):
            _, metrics = fit_on_full_and_save(
                df=df,
                target_column=target_col,
                categorical_cols=cfg["categorical_columns"],
                numeric_cols=cfg["numeric_columns"],
                model_name=model_name,
                model_path=model_path,
            )

        st.success("Модель сохранена")
        st.json(metrics)


# =================================================
# MODEL COMPARISON
# =================================================

def page_model_comparison(df: pd.DataFrame, cfg: dict):

    st.subheader("Сравнение моделей")

    if st.button("Обучить и сравнить"):
        report_df = evaluate_all_models(
            df=df,
            target_column=cfg["default_target_column"],
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=[
                "LogisticRegression",
                "KNN",
                "RandomForest",
                "GradientBoosting",
                "ExtraTrees",
                "MLPClassifier"
            ],
        )

        st.dataframe(report_df, use_container_width=True)

        buf = io.StringIO()
        report_df.to_csv(buf, index=False)
        st.download_button(
            "Скачать отчет сравнения",
            data=buf.getvalue(),
            file_name="model_comparison.csv",
            mime="text/csv"
        )


# =================================================
# PREDICTION
# =================================================

def page_prediction(df: pd.DataFrame, cfg: dict):

    st.subheader("Прогноз вероятности замены устройства")

    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error("Отсутствуют признаки: " + ", ".join(missing))
        return

    model_paths = sorted(glob.glob("models/*.pkl"))

    if not model_paths:
        st.warning("Нет сохраненных моделей.")
        return

    st.markdown("### Параметры устройства")

    input_data = {}

    for col in cfg["categorical_columns"]:
        values = sorted(df[col].dropna().astype(str).unique())
        input_data[col] = st.selectbox(col, values)

    for col in cfg["numeric_columns"]:
        default_val = float(pd.to_numeric(df[col], errors="coerce").mean())
        input_data[col] = st.number_input(col, value=default_val)

    input_df = pd.DataFrame([input_data])

    threshold = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05)

    results = []

    for model_path in model_paths:
        model_name = os.path.basename(model_path)

        try:
            model = load_model(model_path)
            _, proba = predict_needs_upgrade(model, input_df, feature_cols)

            p = float(proba[0])
            decision = "ТРЕБУЕТСЯ" if p >= threshold else "НЕ требуется"

            results.append({
                "Модель": model_name,
                "Вероятность": p,
                "Решение": decision
            })
        except Exception as e:
            results.append({
                "Модель": model_name,
                "Вероятность": float("nan"),
                "Решение": str(e)
            })

    res_df = pd.DataFrame(results).sort_values(
        by="Вероятность",
        ascending=False,
        na_position="last"
    )

    st.dataframe(res_df, use_container_width=True)

    st.markdown("### Индикаторы")

    for _, row in res_df.iterrows():
        if pd.notna(row["Вероятность"]):
            st.write(f"**{row['Модель']}** — {row['Решение']}")
            st.progress(int(row["Вероятность"] * 100))
            st.caption(f"{row['Вероятность']*100:.2f}%")


# =================================================
# EDA
# =================================================

def page_eda(df: pd.DataFrame):

    st.subheader("EDA анализ")

    st.pyplot(plot_ticket_counts_by_department(df))
    st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


# =================================================
# MAIN
# =================================================

def main():

    st.set_page_config(
        page_title="Прогноз обновления техники",
        layout="wide"
    )

    st.title("Интеллектуальная система прогнозирования обновления вычислительной техники")

    cfg = load_config()

    st.sidebar.header("Загрузка данных")

    uploaded_file = st.sidebar.file_uploader("CSV файл", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Файл загружен")
    else:
        df = _load_default_data()
        st.sidebar.info("Используется demo dataset")

    page = st.sidebar.radio(
        "Раздел",
        ["Обзор", "Данные", "Обучение", "Сравнение моделей", "EDA", "Прогноз"]
    )

    if page == "Обзор":
        page_overview()
    elif page == "Данные":
        page_data(df, cfg)
    elif page == "Обучение":
        page_training(df, cfg)
    elif page == "Сравнение моделей":
        page_model_comparison(df, cfg)
    elif page == "EDA":
        page_eda(df)
    elif page == "Прогноз":
        page_prediction(df, cfg)


if __name__ == "__main__":
    main()
