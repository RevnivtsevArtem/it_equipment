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

# ОБЯЗАТЕЛЬНО первой Streamlit-командой
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

from sklearn.model_selection import train_test_split


# =================================================
# DEMO DATA
# =================================================

def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


# =================================================
# ОБЗОР
# =================================================

def page_overview() -> None:
    st.subheader("Общая информация о системе")
    st.markdown(
        """
        Данное приложение реализует интеллектуальный сервис
        прогнозирования потребностей в обновлении вычислительной техники
        на основе обращений в службу технической поддержки.
        """
    )


# =================================================
# ДАННЫЕ (как было в оригинале)
# =================================================

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


# =================================================
# ОБУЧЕНИЕ
# =================================================

def page_training(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Обучение и сохранение модели")

    target_col = cfg["default_target_column"]

    model_name = st.selectbox(
        "Выберите модель для обучения",
        ["LogisticRegression", "KNN", "RandomForest",
         "GradientBoosting", "ExtraTrees", "MLPClassifier"],
    )

    model_filename = st.text_input(
        "Имя файла модели", value=f"{model_name.lower()}_full.pkl"
    )
    model_path = f"models/{model_filename}"

    if st.button("Обучить модель на всём датасете и сохранить"):
        with st.spinner("Идёт обучение модели..."):
            _, metrics = fit_on_full_and_save(
                df=df,
                target_column=target_col,
                categorical_cols=cfg["categorical_columns"],
                numeric_cols=cfg["numeric_columns"],
                model_name=model_name,
                model_path=model_path,
            )
        st.success(f"Модель сохранена в `{model_path}`.")
        st.json(metrics)


# =================================================
# ПРОГНОЗ (доработан, но структура сохранена)
# =================================================

def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз необходимости замены устройства")

    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    model_paths = sorted(glob.glob("models/*.pkl"))

    if not model_paths:
        st.warning("В папке models нет сохранённых моделей.")
        return

    model_names = [os.path.basename(p) for p in model_paths]

    selected_model = st.selectbox(
        "Выберите модель (или все модели)",
        ["ВСЕ модели"] + model_names
    )

    st.markdown("### Параметры устройства")

    input_data = {}

    for col in cfg["categorical_columns"]:
        values = sorted(df[col].dropna().astype(str).unique())
        input_data[col] = st.selectbox(col, values)

    for col in cfg["numeric_columns"]:
        default_val = float(pd.to_numeric(df[col], errors="coerce").mean())
        input_data[col] = st.number_input(col, value=default_val)

    input_df = pd.DataFrame([input_data])

    threshold = st.slider(
        "Порог принятия решения (threshold)",
        0.05, 0.95, 0.5, 0.05
    )

    results = []

    paths_to_use = model_paths if selected_model == "ВСЕ модели" \
        else [p for p in model_paths if os.path.basename(p) == selected_model]

    for path in paths_to_use:
        model_name = os.path.basename(path)
        model = load_model(path)
        _, proba = predict_needs_upgrade(
            model=model,
            df_inputs=input_df,
            feature_cols=feature_cols,
        )

        probability = float(proba[0])
        decision = (
            "ТРЕБУЕТСЯ обновление"
            if probability >= threshold
            else "Пока не требуется"
        )

        results.append({
            "Модель": model_name,
            "Вероятность": probability,
            "Решение": decision
        })

    res_df = pd.DataFrame(results).sort_values(
        by="Вероятность", ascending=False
    )

    st.dataframe(res_df)


# =================================================
# СРАВНЕНИЕ МОДЕЛЕЙ (совместимо с evaluate_all_models)
# =================================================

def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Сравнение моделей")

    target_col = cfg["default_target_column"]

    test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42)

    df_split = df.dropna(subset=[target_col]).copy()
    df_split[target_col] = pd.to_numeric(df_split[target_col], errors="coerce").astype(int)

    df_train, df_test = train_test_split(
        df_split,
        test_size=test_size,
        random_state=random_state,
        stratify=df_split[target_col] if df_split[target_col].nunique() > 1 else None
    )

    if st.button("Обучить и сравнить модели"):
        report_df = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=target_col,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=[
                "LogisticRegression", "KNN", "RandomForest",
                "GradientBoosting", "ExtraTrees", "MLPClassifier"
            ]
        )
        st.dataframe(report_df)


# =================================================
# EDA
# =================================================

def page_eda(df: pd.DataFrame) -> None:
    st.subheader("Разведочный анализ данных (EDA)")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_ticket_counts_by_department(df))
    with col2:
        st.pyplot(plot_ticket_counts_by_device_type(df))

    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


# =================================================
# MAIN (полностью как было + info блок как на 2 скрине)
# =================================================

def main() -> None:

    st.title(
        "Интеллектуальная система прогнозирования потребностей "
        "в обновлении вычислительной техники"
    )

    cfg = load_config()

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV с обращениями", type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Данные успешно загружены.")
    else:
        df = _load_default_data()
        st.info("Используется демонстрационный датасет data/sample_tickets.csv.")

    page = st.sidebar.radio(
        "Раздел приложения",
        ["Обзор", "Данные", "Обучение",
         "Сравнение моделей", "Прогноз устройства", "EDA"],
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
