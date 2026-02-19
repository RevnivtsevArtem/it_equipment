# -*- coding: utf-8 -*-
"""
Главное web-приложение Streamlit.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
"""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from src.app_core import (
    load_config,
    load_model,
    predict_needs_upgrade,
)

def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")

def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз необходимости замены устройства")

    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    if df_clean.empty:
        st.error("Нет корректных данных для прогнозирования.")
        return

    selected_index = st.selectbox("Выберите устройство (строку датасета)", df_clean.index)

    st.write("Параметры выбранного устройства:")
    st.dataframe(df_clean.loc[[selected_index]])

    if st.button("Выполнить прогноз по всем моделям"):

        df_input = df_clean.loc[[selected_index]]

        model_names = [
            "LogisticRegression",
            "KNN",
            "RandomForest",
            "GradientBoosting",
            "ExtraTrees",
            "MLPClassifier",
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
                "probability": float(proba[0])
            })

        if not results:
            st.error("Нет сохранённых моделей. Сначала обучите модели.")
            return

        results_df = pd.DataFrame(results)

        st.subheader("Вероятности по каждой модели")
        st.dataframe(results_df)

        avg_proba = results_df["probability"].mean()

        st.subheader("Ансамблевый прогноз (среднее по моделям)")
        st.progress(avg_proba)
        st.markdown(f"### Средняя вероятность: {avg_proba:.2%}")

        if avg_proba >= 0.7:
            st.error("Высокая вероятность необходимости замены устройства")
        elif avg_proba >= 0.5:
            st.warning("Средняя вероятность необходимости замены устройства")
        else:
            st.success("Низкая вероятность необходимости замены устройства")

def main() -> None:
    st.set_page_config(page_title="Прогноз обновления вычислительной техники", layout="wide")
    st.title("Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники")

    cfg = load_config()

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV с обращениями", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Данные успешно загружены.")
    else:
        df = _load_default_data()
        st.info("Используется демонстрационный датасет `data/sample_tickets.csv`.")

    page_prediction(df, cfg)

if __name__ == "__main__":
    main()
