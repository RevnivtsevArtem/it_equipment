# -*- coding: utf-8 -*-
"""
Главное web-приложение Streamlit.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.app_core import (
    load_config,
    fit_on_full_and_save,
    load_model,
    predict_needs_upgrade,
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


def _load_default_data() -> pd.DataFrame:
    """Загружает демонстрационный датасет."""
    return pd.read_csv("data/sample_tickets.csv")


def page_overview() -> None:
    """Вкладка с кратким описанием системы."""
    st.subheader("Общая информация о системе")
    st.markdown(
        """
        Данное приложение реализует интеллектуальный сервис прогнозирования потребностей
        в обновлении вычислительной техники на основе обращений в службу технической поддержки.

        Основные функции:
        - загрузка и просмотр данных;
        - обучение и сохранение моделей;
        - применение обученных моделей для прогнозирования;
        - сравнение нескольких моделей по ключевым метрикам качества;
        - визуальный анализ данных и результатов обучения.
        """
    )


def page_data(df: pd.DataFrame, cfg: dict) -> None:
    """Вкладка для работы с данными."""
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
    """Вкладка обучения модели на всём датасете (демонстрационный режим)."""
    st.subheader("Обучение и сохранение модели")
    target_col = cfg["default_target_column"]
    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует в данных.")
        return

    model_name = st.selectbox(
        "Выберите модель для обучения",
        ["LogisticRegression", "KNN", "RandomForest", "GradientBoosting", "ExtraTrees", "MLPClassifier"],
    )
    model_filename = st.text_input("Имя файла модели", value=f"{model_name.lower()}_full.pkl")
    model_path = f"models/{model_filename}"

    if st.button("Обучить модель на всём датасете и сохранить"):
        with st.spinner("Идёт обучение модели..."):
            pipeline, metrics = fit_on_full_and_save(
                df=df,
                target_column=target_col,
                categorical_cols=cfg["categorical_columns"],
                numeric_cols=cfg["numeric_columns"],
                model_name=model_name,
                model_path=model_path,
            )
        st.success(f"Модель `{model_name}` сохранена в `{model_path}`.")
        st.json(metrics)


def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    """Вкладка сравнения нескольких моделей на train/test разбиении."""
    st.subheader("Сравнение моделей")
    target_col = cfg["default_target_column"]
    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует в данных.")
        return

    from sklearn.model_selection import train_test_split

    test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col].astype(int).values,
    )

    models_to_compare = st.multiselect(
        "Выберите модели для сравнения",
        ["LogisticRegression", "KNN", "RandomForest", "GradientBoosting", "ExtraTrees", "MLPClassifier"],
        default=["LogisticRegression", "RandomForest", "GradientBoosting"],
    )

    if st.button("Обучить и сравнить модели"):
        metrics_all = {}
        for name in models_to_compare:
            with st.spinner(f"Обучение модели {name}..."):
                _, metrics = fit_model_and_evaluate(
                    df_train=df_train,
                    df_test=df_test,
                    target_column=target_col,
                    categorical_cols=cfg["categorical_columns"],
                    numeric_cols=cfg["numeric_columns"],
                    model_name=name,
                )
            metrics_all[name] = metrics

        st.markdown("### Итоговая таблица метрик")
        md = metrics_to_markdown_table(metrics_all)
        st.markdown(md)


def page_eda(df: pd.DataFrame) -> None:
    """Вкладка EDA."""
    st.subheader("Разведочный анализ данных (EDA)")
    st.write("Всего записей:", len(df))

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_ticket_counts_by_department(df))
    with col2:
        st.pyplot(plot_ticket_counts_by_device_type(df))

    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


def page_report(df: pd.DataFrame, cfg: dict) -> None:
    """Вкладка генерации текстового отчёта по выбранной сохранённой модели."""
    st.subheader("Демонстрация сохранённой модели и отчёт")
    model_filename = st.text_input("Имя файла сохранённой модели", value="randomforest_full.pkl")
    model_path = f"models/{model_filename}"

    target_col = cfg["default_target_column"]
    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    if st.button("Загрузить модель и выполнить оценку"):
        try:
            model = load_model(model_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

        df_sample = df.copy()
        df_sample = df_sample.dropna(subset=[target_col])
        X = df_sample[feature_cols]
        y_true = df_sample[target_col].astype(int).values

        labels, proba = predict_needs_upgrade(model, df_sample, feature_cols)

        from src.evaluation import evaluate_binary_classifier

        metrics = evaluate_binary_classifier(y_true, labels, proba)
        st.markdown("### Метрики модели")
        st.json(metrics)

        st.markdown("### Матрица ошибок")
        st.pyplot(plot_confusion_matrix(y_true, labels))

        if proba is not None:
            st.markdown("### ROC-кривая")
            st.pyplot(plot_roc_curve(y_true, proba))

            st.markdown("### PR-кривая")
            st.pyplot(plot_pr_curve(y_true, proba))

        st.markdown("### Пример прогноза по первым 20 записям")
        df_preview = df_sample.head(20).copy()
        df_preview["pred_label"] = labels[:20]
        df_preview["pred_proba"] = proba[:20]
        st.dataframe(df_preview)

        buf = io.StringIO()
        df_preview.to_csv(buf, index=False)
        st.download_button(
            "Скачать пример прогноза (CSV)",
            data=buf.getvalue(),
            file_name="model_report_sample_predictions.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="Прогноз обновления вычислительной техники", layout="wide")
    st.title("Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники")

    st.markdown(
        """
        **Автор ВКР:** Ревнивцев Артём Александрович  
        **Тема:** Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники  
        на основе обращений в службу технической поддержки  
        (на примере ЧОУ ВО «Московский университет имени С.Ю. Витте»)
        """
    )

    cfg = load_config()

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV с обращениями", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Данные успешно загружены.")
    else:
        df = _load_default_data()
        st.info("Используется демонстрационный датасет `data/sample_tickets.csv`.")

    page = st.sidebar.radio(
        "Раздел приложения",
        ["Обзор", "Данные", "Обучение", "Сравнение моделей", "EDA", "Отчёт"],
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
    elif page == "Отчёт":
        page_report(df, cfg)


if __name__ == "__main__":
    main()