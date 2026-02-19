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

# ---------------------------------------------------
# НАСТРОЙКА СТРАНИЦЫ (обязательно первой командой)
# ---------------------------------------------------
st.set_page_config(
    page_title="Прогноз обновления вычислительной техники",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------
# ИМПОРТ МОДУЛЕЙ ПРОЕКТА
# ---------------------------------------------------
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

# ---------------------------------------------------
# ЗАГРУЗКА ДЕМО ДАТАСЕТА
# ---------------------------------------------------
def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")

# ---------------------------------------------------
# СТРАНИЦА 1 — ОБЗОР
# ---------------------------------------------------
def page_overview():
    st.subheader("Общая информация о системе")

    st.markdown("""
Данное приложение реализует интеллектуальный сервис прогнозирования потребностей
в обновлении вычислительной техники на основе обращений в службу технической поддержки.

Основные возможности:
- загрузка и просмотр данных
- обучение моделей
- сравнение алгоритмов
- прогноз вероятности обновления
- формирование итогового отчёта
    """)

# ---------------------------------------------------
# СТРАНИЦА 2 — ДАННЫЕ
# ---------------------------------------------------
def page_data(df, cfg):
    st.subheader("Работа с данными")
    st.write("Размер набора данных:", df.shape)

    st.dataframe(df.head(20), use_container_width=True)

    if st.checkbox("Показать статистику числовых признаков"):
        st.write(df[cfg["numeric_columns"]].describe())

    if st.checkbox("Скачать текущий датасет"):
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)

        st.download_button(
            label="Скачать CSV",
            data=buffer.getvalue(),
            file_name="dataset.csv",
            mime="text/csv",
        )

# ---------------------------------------------------
# СТРАНИЦА 3 — ОБУЧЕНИЕ
# ---------------------------------------------------
def page_training(df, cfg):
    st.subheader("Обучение и сохранение модели")

    target_col = cfg["default_target_column"]

    model_name = st.selectbox(
        "Выберите модель",
        [
            "LogisticRegression",
            "KNN",
            "RandomForest",
            "GradientBoosting",
            "ExtraTrees",
            "MLPClassifier",
        ],
    )

    model_filename = st.text_input(
        "Имя файла модели",
        value=f"{model_name.lower()}_model.pkl"
    )

    model_path = f"models/{model_filename}"

    if st.button("Обучить и сохранить модель"):
        with st.spinner("Идёт обучение..."):
            _, metrics = fit_on_full_and_save(
                df=df,
                target_column=target_col,
                categorical_cols=cfg["categorical_columns"],
                numeric_cols=cfg["numeric_columns"],
                model_name=model_name,
                model_path=model_path,
            )

        st.success(f"Модель сохранена в {model_path}")
        st.json(metrics)

# ---------------------------------------------------
# СТРАНИЦА 4 — ПРОГНОЗ
# ---------------------------------------------------
def page_prediction(df, cfg):
    st.subheader("Прогноз необходимости замены устройства")

    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    model_paths = sorted(glob.glob("models/*.pkl"))

    if not model_paths:
        st.warning("В папке models нет обученных моделей.")
        return

    model_names = [os.path.basename(p) for p in model_paths]

    selected_model = st.selectbox(
        "Выберите модель",
        ["ВСЕ модели"] + model_names
    )

    st.markdown("### Параметры устройства")

    input_data = {}

    for col in cfg["categorical_columns"]:
        values = sorted(df[col].astype(str).unique())
        input_data[col] = st.selectbox(col, values)

    for col in cfg["numeric_columns"]:
        default_val = float(df[col].mean())
        input_data[col] = st.number_input(col, value=default_val)

    threshold = st.slider(
        "Порог принятия решения",
        0.1, 0.9, 0.5, 0.05
    )

    if st.button("Спрогнозировать"):

        paths = model_paths if selected_model == "ВСЕ модели" \
                else [p for p in model_paths if os.path.basename(p) == selected_model]

        results = []

        for p in paths:
            name = os.path.basename(p)
            model = load_model(p)

            _, proba = predict_needs_upgrade(
                model=model,
                df_inputs=pd.DataFrame([input_data]),
                feature_cols=feature_cols,
            )

            prob = float(proba[0])
            decision = "ТРЕБУЕТСЯ обновление" if prob >= threshold else "Пока не требуется"

            results.append({
                "Модель": name,
                "Вероятность": prob,
                "Решение": decision
            })

        res_df = pd.DataFrame(results).sort_values(by="Вероятность", ascending=False)

        st.markdown("## Итог")
        best = res_df.iloc[0]

        st.metric(
            "Максимальная вероятность обновления",
            f"{best['Вероятность']*100:.2f}%"
        )

        if best["Вероятность"] >= threshold:
            st.error(best["Решение"])
        else:
            st.success(best["Решение"])

        st.dataframe(res_df, use_container_width=True)

# ---------------------------------------------------
# СТРАНИЦА 5 — СРАВНЕНИЕ МОДЕЛЕЙ
# ---------------------------------------------------
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

    if st.button("Обучить и сравнить модели"):

        report = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=cfg["default_target_column"],
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

        st.dataframe(report, use_container_width=True)

# ---------------------------------------------------
# СТРАНИЦА 6 — EDA
# ---------------------------------------------------
def page_eda(df):
    st.subheader("Разведочный анализ данных (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_ticket_counts_by_department(df))

    with col2:
        st.pyplot(plot_ticket_counts_by_device_type(df))

    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))

# ---------------------------------------------------
# СТРАНИЦА 7 — ОТЧЁТ
# ---------------------------------------------------
def page_report(df, cfg):
    from sklearn.model_selection import train_test_split

    st.subheader("Итоговый отчёт")

    test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2, 0.05)

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[cfg["default_target_column"]],
    )

    model_name = st.selectbox(
        "Выберите модель для детальной оценки",
        [
            "LogisticRegression",
            "RandomForest",
            "GradientBoosting",
            "ExtraTrees",
            "MLPClassifier",
        ],
    )

    if st.button("Построить отчёт"):
        eval_out = fit_model_and_evaluate(
            df_train,
            df_test,
            cfg["default_target_column"],
            cfg["categorical_columns"],
            cfg["numeric_columns"],
            model_name,
        )

        pipeline, metrics, y_true, y_pred, y_proba = eval_out

        st.markdown("### Метрики")
        st.markdown(metrics_to_markdown_table(metrics))

        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_confusion_matrix(y_true, y_pred))

        with col2:
            st.pyplot(plot_roc_curve(y_true, y_proba))

        st.pyplot(plot_pr_curve(y_true, y_proba))

# ---------------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ
# ---------------------------------------------------
def main():

    st.title(
        "Интеллектуальная система прогнозирования "
        "потребностей в обновлении вычислительной техники"
    )

    st.markdown("""
**Автор ВКР:** Ревнивцев Артём Александрович  
**Тема:** Интеллектуальная система прогнозирования потребностей
в обновлении вычислительной техники
на основе обращений в службу технической поддержки
""")

    cfg = load_config()

    # ---------- SIDEBAR ----------
    st.sidebar.header("Загрузка данных")

    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV с обращениями",
        type=["csv"],
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Данные загружены")
    else:
        df = _load_default_data()
        st.sidebar.info("Используется демонстрационный датасет")

    st.sidebar.divider()

    st.sidebar.header("Раздел приложения")

    page = st.sidebar.radio(
        "Выберите раздел",
        [
            "Обзор",
            "Данные",
            "Обучение",
            "Сравнение моделей",
            "Прогноз",
            "EDA",
            "Отчёт",
        ],
    )

    # ---------- РЕНДЕР СТРАНИЦ ----------
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

    elif page == "Отчёт":
        page_report(df, cfg)


if __name__ == "__main__":
    main()
