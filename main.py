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
# ДАННЫЕ
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

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует в данных.")
        return

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
# ПРОГНОЗ (ДОРАБОТАННЫЙ ПО ВСЕМ ТРЕБОВАНИЯМ)
# =================================================

def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз необходимости замены устройства")

    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    # --- 1. Автопоиск моделей ---
    model_paths = sorted(glob.glob("models/*.pkl"))

    if not model_paths:
        st.warning("В папке models нет сохранённых моделей. Сначала обучите модель.")
        return

    model_names = [os.path.basename(p) for p in model_paths]

    # --- 2. Выбор модели (выпадающий список) ---
    selected_model = st.selectbox(
        "Выберите модель (или используйте все модели)",
        ["ВСЕ модели"] + model_names
    )

    # --- 3. Параметры устройства ---
    st.markdown("### Параметры устройства")

    input_data = {}

    for col in cfg["categorical_columns"]:
        values = sorted(df[col].dropna().astype(str).unique())
        input_data[col] = st.selectbox(col, values)

    for col in cfg["numeric_columns"]:
        default_val = float(pd.to_numeric(df[col], errors="coerce").mean())
        input_data[col] = st.number_input(col, value=default_val)

    input_df = pd.DataFrame([input_data])

    # --- 4. Threshold ---
    threshold = st.slider(
        "Порог принятия решения (threshold)",
        0.05, 0.95, 0.5, 0.05
    )

    st.info(
        f"Если вероятность ≥ {threshold:.2f}, устройство считается требующим обновления."
    )

    # --- 5. Расчёт ---
    results = []

    paths_to_use = model_paths if selected_model == "ВСЕ модели" \
        else [p for p in model_paths if os.path.basename(p) == selected_model]

    for path in paths_to_use:
        model_name = os.path.basename(path)

        try:
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

        except Exception as e:
            results.append({
                "Модель": model_name,
                "Вероятность": None,
                "Решение": f"Ошибка: {e}"
            })

    res_df = pd.DataFrame(results).sort_values(
        by="Вероятность", ascending=False, na_position="last"
    )

    st.markdown("## Результаты прогнозирования")
    st.dataframe(res_df, use_container_width=True)

    # --- Индикаторы ---
    st.markdown("### Индикаторы вероятности")
    for _, row in res_df.iterrows():
        if row["Вероятность"] is None:
            continue
        st.write(f"**{row['Модель']}** — {row['Решение']}")
        st.progress(int(row["Вероятность"] * 100))
        st.caption(f"{row['Вероятность']*100:.2f}%")


# =================================================
# СРАВНЕНИЕ МОДЕЛЕЙ (СОХРАНЕНО)
# =================================================

def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Сравнение моделей")

    if st.button("Обучить и сравнить модели"):
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
        st.dataframe(report_df)


# =================================================
# EDA (СОХРАНЕНО)
# =================================================

def page_eda(df: pd.DataFrame) -> None:
    st.subheader("Разведочный анализ данных (EDA)")
    st.pyplot(plot_ticket_counts_by_department(df))
    st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


# =================================================
# MAIN (ОРИГИНАЛЬНАЯ СТРУКТУРА СОХРАНЕНА)
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
        st.sidebar.success("Данные успешно загружены.")
    else:
        df = _load_default_data()
        st.sidebar.info("Используется демонстрационный датасет.")

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
