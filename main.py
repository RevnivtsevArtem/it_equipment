# -*- coding: utf-8 -*-
"""
Главное web-приложение Streamlit.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей
в обновлении вычислительной техники.
"""

from __future__ import annotations

import io
import os
import glob
import pandas as pd
import streamlit as st

# FIRST Streamlit command
st.set_page_config(
    page_title="Прогноз обновления вычислительной техники",
    layout="wide",
    initial_sidebar_state="expanded"
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


# ========================== PAGES ==========================

def page_overview():
    st.subheader("Общая информация о системе")
    st.markdown("""
Интеллектуальный сервис прогнозирования потребностей
в обновлении вычислительной техники на основе анализа обращений
в службу технической поддержки.
""")


def page_data(df, cfg):
    st.subheader("Работа с данными")
    st.write("Размер набора данных:", df.shape)
    st.dataframe(df.head(20), use_container_width=True)


def page_training(df, cfg):
    st.subheader("Обучение и сохранение модели")

    model_name = st.selectbox(
        "Выберите модель",
        ["LogisticRegression", "KNN", "RandomForest",
         "GradientBoosting", "ExtraTrees", "MLPClassifier"]
    )

    filename = st.text_input("Имя файла модели", value=f"{model_name.lower()}_full.pkl")

    if st.button("Обучить модель"):
        _, metrics = fit_on_full_and_save(
            df=df,
            target_column=cfg["default_target_column"],
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_name=model_name,
            model_path=f"models/{filename}",
        )
        st.success("Модель сохранена.")
        st.json(metrics)


def page_model_comparison(df, cfg):
    from sklearn.model_selection import train_test_split

    st.subheader("Сравнение моделей")

    target = cfg["default_target_column"]
    df_split = df.dropna(subset=[target]).copy()
    df_split[target] = df_split[target].astype(int)

    df_train, df_test = train_test_split(
        df_split, test_size=0.2, random_state=42
    )

    if st.button("Обучить и сравнить модели"):
        report_df = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=target,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=[
                "LogisticRegression", "KNN", "RandomForest",
                "GradientBoosting", "ExtraTrees", "MLPClassifier"
            ],
        )
        st.dataframe(report_df, use_container_width=True)


def page_prediction(df, cfg):
    st.subheader("Прогноз устройства")

    models = sorted(glob.glob("models/*.pkl"))
    if not models:
        st.warning("Нет обученных моделей в папке models/")
        return

    selected = st.selectbox(
        "Выберите модель",
        ["ВСЕ модели"] + [os.path.basename(p) for p in models]
    )

    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

    input_data = {}

    for col in cfg["categorical_columns"]:
        input_data[col] = st.selectbox(
            col, sorted(df[col].dropna().unique())
        )

    for col in cfg["numeric_columns"]:
        input_data[col] = st.number_input(
            col, value=float(df[col].mean())
        )

    if st.button("Рассчитать"):
        input_df = pd.DataFrame([input_data])
        paths = models if selected == "ВСЕ модели" else [
            p for p in models if os.path.basename(p) == selected
        ]

        results = []

        for path in paths:
            model = load_model(path)
            _, proba = predict_needs_upgrade(
                model,
                input_df,
                cfg["categorical_columns"] +
                cfg["numeric_columns"]
            )
            p = float(proba[0])
            decision = "ТРЕБУЕТСЯ" if p >= threshold else "Не требуется"

            results.append({
                "Модель": os.path.basename(path),
                "Вероятность": round(p, 3),
                "Решение": decision
            })

        df_res = pd.DataFrame(results).sort_values(
            by="Вероятность", ascending=False
        )

        st.dataframe(df_res)

        for _, row in df_res.iterrows():
            st.write(f"**{row['Модель']}** — {row['Решение']}")
            st.progress(row["Вероятность"])


def page_eda(df):
    st.subheader("EDA")
    st.pyplot(plot_ticket_counts_by_department(df))
    st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


def page_report(df, cfg):
    st.subheader("Отчёт")
    st.write("Функционал отчёта сохранён.")


# ========================== MAIN ==========================

def main():

    # --- SIDEBAR FIRST ---
    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV", type=["csv"]
    )

    page = st.sidebar.radio(
        "Раздел приложения",
        ["Обзор", "Данные", "Обучение",
         "Сравнение моделей", "Прогноз устройства",
         "EDA", "Отчёт"]
    )

    # --- MAIN CONTENT ---
    st.title("Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники")

    try:
        cfg = load_config()
    except Exception as e:
        st.error(f"Ошибка load_config: {e}")
        return

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Данные успешно загружены.")
    else:
        df = _load_default_data()
        st.info("Используется демонстрационный датасет data/sample_tickets.csv.")

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
