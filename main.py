# -*- coding: utf-8 -*-
"""
Главное web-приложение Streamlit.

Автор: Ревнивцев Артём Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей
в обновлении вычислительной техники.
"""

from __future__ import annotations

import io
import os
import glob
import pandas as pd
import streamlit as st

# --- НАСТРОЙКА СТРАНИЦЫ ---
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
    evaluate_all_models,
    fit_model_and_evaluate,
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


# --- ЗАГРУЗКА ДЕМО ДАТАСЕТА ---
def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


# --- СТРАНИЦЫ ---
def page_overview():
    st.subheader("Общая информация о системе")
    st.markdown("""
    Система предназначена для интеллектуального прогнозирования потребности
    в обновлении вычислительной техники на основе обращений в службу технической поддержки.

    Реализованы:
    - анализ данных;
    - обучение моделей;
    - сравнение алгоритмов;
    - прогнозирование вероятности замены устройства;
    - формирование итогового отчёта.
    """)


def page_data(df, cfg):
    st.subheader("Работа с данными")
    st.write("Размер набора данных:", df.shape)
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
            mime="text/csv",
        )


def page_training(df, cfg):
    st.subheader("Обучение и сохранение модели")

    model_name = st.selectbox(
        "Выберите модель",
        ["LogisticRegression", "RandomForest", "GradientBoosting", "ExtraTrees", "MLPClassifier"],
    )

    if st.button("Обучить и сохранить"):
        with st.spinner("Обучение модели..."):
            _, metrics = fit_on_full_and_save(
                df=df,
                target_column=cfg["default_target_column"],
                categorical_cols=cfg["categorical_columns"],
                numeric_cols=cfg["numeric_columns"],
                model_name=model_name,
                model_path=f"models/{model_name.lower()}_model.pkl",
            )
        st.success("Модель успешно обучена и сохранена.")
        st.json(metrics)


def page_prediction(df, cfg):
    st.subheader("Прогноз устройства")

    model_paths = sorted(glob.glob("models/*.pkl"))
    if not model_paths:
        st.warning("Сначала обучите модель.")
        return

    model_names = [os.path.basename(p) for p in model_paths]
    selected_model = st.selectbox("Выберите модель", model_names)

    input_data = {}

    st.markdown("### Параметры устройства")

    for col in cfg["categorical_columns"]:
        input_data[col] = st.selectbox(col, sorted(df[col].astype(str).unique()))

    for col in cfg["numeric_columns"]:
        default_val = float(pd.to_numeric(df[col], errors="coerce").mean())
        input_data[col] = st.number_input(col, value=default_val)

    threshold = st.slider("Порог принятия решения", 0.1, 0.9, 0.5, 0.05)

    if st.button("Спрогнозировать"):
        model = load_model(f"models/{selected_model}")
        _, proba = predict_needs_upgrade(
            model=model,
            df_inputs=pd.DataFrame([input_data]),
            feature_cols=cfg["categorical_columns"] + cfg["numeric_columns"],
        )

        probability = float(proba[0])

        st.markdown("## Результат")
        st.metric("Вероятность обновления", f"{probability * 100:.2f}%")

        if probability >= threshold:
            st.error("ТРЕБУЕТСЯ обновление устройства")
        else:
            st.success("Обновление пока не требуется")


def page_model_comparison(df, cfg):
    from sklearn.model_selection import train_test_split

    st.subheader("Сравнение моделей")

    test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2, 0.05)

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[cfg["default_target_column"]],
    )

    if st.button("Обучить и сравнить"):
        report = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=cfg["default_target_column"],
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=[
                "LogisticRegression",
                "RandomForest",
                "GradientBoosting",
                "ExtraTrees",
                "MLPClassifier",
            ],
        )

        st.dataframe(report, use_container_width=True)


def page_eda(df):
    st.subheader("Разведочный анализ данных (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_ticket_counts_by_department(df))
    with col2:
        st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


def main():
    st.title("Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники")

    st.markdown("""
    **Автор ВКР:** Ревнивцев Артём Александрович  
    **Тема:** Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники
    """)

    cfg = load_config()

    # --- SIDEBAR ---
    st.sidebar.markdown("## Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Данные загружены")
    else:
        df = _load_default_data()
        st.sidebar.info("Используется демонстрационный датасет")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Раздел приложения")

    page = st.sidebar.radio(
        "",
        ["Обзор", "Данные", "Обучение", "Сравнение моделей", "Прогноз", "EDA"],
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
    elif page == "EDA":
        page_eda(df)


if __name__ == "__main__":
    main()
