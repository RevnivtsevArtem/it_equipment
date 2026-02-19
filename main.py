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
    evaluate_all_models,
)

from src.eda import (
    plot_ticket_counts_by_department,
    plot_ticket_counts_by_device_type,
    plot_device_age_hist,
    plot_tickets_last_6_months_hist,
)


# ==============================
# DEMO DATA
# ==============================

def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


# ==============================
# ОБЗОР
# ==============================

def page_overview() -> None:
    st.subheader("Общая информация о системе")
    st.markdown(
        """
        Интеллектуальный сервис прогнозирования потребностей
        в обновлении вычислительной техники на основе обращений в службу технической поддержки.
        """
    )


# ==============================
# ДАННЫЕ
# ==============================

def page_data(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Работа с данными")
    st.write("Размер набора данных:", df.shape)
    st.dataframe(df.head(20), use_container_width=True)

    if st.checkbox("Показать статистику числовых признаков"):
        st.write(df[cfg["numeric_columns"]].describe())

    if st.checkbox("Скачать текущий датасет"):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Скачать CSV", data=buf.getvalue(),
                           file_name="dataset.csv", mime="text/csv")


# ==============================
# ОБУЧЕНИЕ
# ==============================

def page_training(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Обучение и сохранение модели")

    target_col = cfg["default_target_column"]

    model_name = st.selectbox(
        "Выберите модель для обучения",
        ["LogisticRegression", "KNN", "RandomForest",
         "GradientBoosting", "ExtraTrees", "MLPClassifier"],
    )

    model_filename = st.text_input(
        "Имя файла модели", value=f"{model_name.lower()}_full.pkl")
    model_path = f"models/{model_filename}"

    if st.button("Обучить и сохранить"):
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


# ==============================
# СРАВНЕНИЕ МОДЕЛЕЙ
# ==============================

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
        st.dataframe(report_df, use_container_width=True)


# ==============================
# ПРОГНОЗ (полностью доработанный)
# ==============================

def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз необходимости замены устройства")

    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    # Автопоиск всех моделей
    model_paths = sorted(glob.glob("models/*.pkl"))

    if not model_paths:
        st.warning("В папке models нет сохранённых моделей.")
        return

    st.caption("Найденные модели:")
    st.code("\n".join(model_paths))

    # Автоподстановка категорий
    st.markdown("### Параметры устройства")
    input_data = {}

    for col in cfg["categorical_columns"]:
        values = sorted(df[col].dropna().astype(str).unique())
        input_data[col] = st.selectbox(col, values)

    for col in cfg["numeric_columns"]:
        default_val = float(pd.to_numeric(df[col], errors="coerce").mean())
        input_data[col] = st.number_input(col, value=default_val)

    input_df = pd.DataFrame([input_data])

    # Threshold
    threshold = st.slider(
        "Порог принятия решения (threshold)",
        0.05, 0.95, 0.5, 0.05
    )

    st.info(
        f"Если вероятность ≥ {threshold:.2f}, устройство считается требующим обновления."
    )

    # Расчёт сразу по всем моделям
    results = []

    for path in model_paths:
        model_name = os.path.basename(path)
        try:
            model = load_model(path)
            _, proba = predict_needs_upgrade(model, input_df, feature_cols)

            probability = float(proba[0])
            decision = "ТРЕБУЕТСЯ" if probability >= threshold else "НЕ требуется"

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

    st.dataframe(res_df, use_container_width=True)

    st.markdown("### Индикаторы вероятности")
    for _, row in res_df.iterrows():
        if row["Вероятность"] is None:
            continue

        st.write(f"**{row['Модель']}** — {row['Решение']}")
        st.progress(int(row["Вероятность"] * 100))
        st.caption(f"{row['Вероятность']*100:.2f}%")


# ==============================
# EDA
# ==============================

def page_eda(df: pd.DataFrame) -> None:
    st.subheader("Разведочный анализ данных (EDA)")
    st.pyplot(plot_ticket_counts_by_department(df))
    st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


# ==============================
# MAIN
# ==============================

def main() -> None:

    st.title(
        "Интеллектуальная система прогнозирования потребностей "
        "в обновлении вычислительной техники"
    )

    cfg = load_config()

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV с обращениями", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Данные загружены.")
    else:
        df = _load_default_data()
        st.sidebar.info("Используется demo dataset.")

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
