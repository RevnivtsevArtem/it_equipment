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


# =========================================================
# Загрузка демо-датасета
# =========================================================

def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


# =========================================================
# Страницы приложения
# =========================================================

def page_overview() -> None:
    st.subheader("Общая информация о системе")
    st.markdown(
        """
Данное приложение реализует интеллектуальный сервис прогнозирования
потребностей в обновлении вычислительной техники на основе обращений
в службу технической поддержки.

Основные функции:
- загрузка и просмотр данных;
- обучение и сохранение моделей;
- применение обученных моделей;
- сравнение моделей;
- визуальный анализ данных.
"""
    )


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
    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует.")
        return

    model_name = st.selectbox(
        "Выберите модель",
        ["LogisticRegression", "KNN", "RandomForest",
         "GradientBoosting", "ExtraTrees", "MLPClassifier"],
    )

    model_filename = st.text_input(
        "Имя файла модели",
        value=f"{model_name.lower()}_full.pkl"
    )

    model_path = f"models/{model_filename}"

    if st.button("Обучить модель и сохранить"):
        with st.spinner("Идёт обучение модели..."):
            pipeline, metrics = fit_on_full_and_save(
                df=df,
                target_column=target_col,
                categorical_cols=cfg["categorical_columns"],
                numeric_cols=cfg["numeric_columns"],
                model_name=model_name,
                model_path=model_path,
            )
        st.success(f"Модель сохранена в {model_path}")
        st.json(metrics)


def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз необходимости замены устройства")

    target_col = cfg["default_target_column"]
    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует.")
        return

    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    if df_clean.empty:
        st.error("Нет корректных данных.")
        return

    selected_index = st.selectbox(
        "Выберите строку датасета",
        df_clean.index
    )

    st.dataframe(df_clean.loc[[selected_index]])

    if st.button("Спрогнозировать по всем моделям"):

        df_input = df_clean.loc[[selected_index]]

        model_names = [
            "LogisticRegression", "KNN", "RandomForest",
            "GradientBoosting", "ExtraTrees", "MLPClassifier"
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
            st.error("Нет обученных моделей.")
            return

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        avg_proba = results_df["probability"].mean()

        st.progress(avg_proba)
        st.markdown(f"### Средняя вероятность: {avg_proba:.2%}")


def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    from sklearn.model_selection import train_test_split

    st.subheader("Сравнение моделей")

    target_col = cfg["default_target_column"]

    if target_col not in df.columns:
        st.error("Отсутствует целевой столбец.")
        return

    test_size = st.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42)

    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean[target_col] = df_clean[target_col].astype(int)

    df_train, df_test = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state,
        stratify=df_clean[target_col] if df_clean[target_col].nunique() > 1 else None
    )

    models = [
        "LogisticRegression", "KNN", "RandomForest",
        "GradientBoosting", "ExtraTrees", "MLPClassifier"
    ]

    if st.button("Обучить и сравнить"):
        report_df = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=target_col,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=models,
        )

        st.dataframe(report_df)


def page_eda(df: pd.DataFrame) -> None:
    st.subheader("EDA")
    st.pyplot(plot_ticket_counts_by_department(df))
    st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    st.set_page_config(
        page_title="Прогноз обновления техники",
        layout="wide"
    )

    st.title("Интеллектуальная система прогнозирования")

    cfg = load_config()

    # ============================
    # ДОРАБОТАННАЯ ЗАГРУЗКА ДАННЫХ
    # ============================

    st.sidebar.header("Загрузка данных")

    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV",
        type=["csv"]
    )

    required_columns = (
        cfg["categorical_columns"]
        + cfg["numeric_columns"]
        + [cfg["default_target_column"]]
    )

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)

            missing_cols = [
                col for col in required_columns
                if col not in df_uploaded.columns
            ]

            if missing_cols:
                st.sidebar.error(
                    f"Отсутствуют столбцы: {missing_cols}"
                )
                st.stop()

            st.session_state["current_df"] = df_uploaded
            st.sidebar.success("Пользовательский датасет загружен")

        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки: {e}")
            st.stop()

    if "current_df" not in st.session_state:
        st.session_state["current_df"] = _load_default_data()
        st.sidebar.info("Используется демонстрационный датасет")

    df = st.session_state["current_df"]

    page = st.sidebar.radio(
        "Раздел",
        ["Обзор", "Данные", "Обучение",
         "Сравнение моделей", "Прогноз устройства", "EDA"]
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


if __name__ == "__main__":
    main()
