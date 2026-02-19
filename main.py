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


# ====================== DATA ======================

def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


# ====================== PAGES ======================

def page_overview() -> None:
    st.subheader("Общая информация о системе")
    st.markdown("""
Интеллектуальный сервис прогнозирования потребностей
в обновлении вычислительной техники на основе анализа
обращений в службу технической поддержки.
""")


def page_data(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Работа с данными")
    st.write("Размер набора данных:", df.shape)
    st.dataframe(df.head(20))

    if st.checkbox("Показать статистику числовых признаков"):
        st.write(df[cfg["numeric_columns"]].describe())

    if st.checkbox("Скачать текущий датасет в CSV"):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Скачать CSV", buf.getvalue(),
                           file_name="dataset.csv", mime="text/csv")


def page_training(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Обучение и сохранение модели")

    model_name = st.selectbox(
        "Выберите модель",
        ["LogisticRegression", "KNN", "RandomForest",
         "GradientBoosting", "ExtraTrees", "MLPClassifier"]
    )

    model_filename = st.text_input("Имя файла модели",
                                   value=f"{model_name.lower()}_full.pkl")

    if st.button("Обучить модель"):
        pipeline, metrics = fit_on_full_and_save(
            df=df,
            target_column=cfg["default_target_column"],
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_name=model_name,
            model_path=f"models/{model_filename}",
        )
        st.success("Модель сохранена.")
        st.json(metrics)


def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    from sklearn.model_selection import train_test_split

    st.subheader("Сравнение моделей")

    target_col = cfg["default_target_column"]

    df_split = df.dropna(subset=[target_col]).copy()
    df_split[target_col] = df_split[target_col].astype(int)

    df_train, df_test = train_test_split(
        df_split, test_size=0.2, random_state=42
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
            ],
        )
        st.dataframe(report_df)


def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз устройства")

    models = glob.glob("models/*.pkl")
    if not models:
        st.warning("Нет обученных моделей в папке models/")
        return

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

        results = []
        for model_path in models:
            model = load_model(model_path)
            _, proba = predict_needs_upgrade(
                model,
                input_df,
                cfg["categorical_columns"] +
                cfg["numeric_columns"]
            )
            p = float(proba[0])
            decision = "ТРЕБУЕТСЯ" if p >= threshold else "Не требуется"
            results.append({
                "Модель": os.path.basename(model_path),
                "Вероятность": round(p, 3),
                "Решение": decision
            })

        res_df = pd.DataFrame(results).sort_values(
            by="Вероятность", ascending=False)

        st.dataframe(res_df)

        for _, row in res_df.iterrows():
            st.write(f"**{row['Модель']}** — {row['Решение']}")
            st.progress(row["Вероятность"])


def page_eda(df: pd.DataFrame) -> None:
    st.subheader("EDA")
    st.pyplot(plot_ticket_counts_by_department(df))
    st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


def page_report(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Отчёт")
    st.write("Функционал итогового отчёта сохранён.")


# ====================== MAIN ======================

def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники")

    cfg = load_config()

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV", type=["csv"]
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
         "Сравнение моделей", "Прогноз устройства",
         "EDA", "Отчёт"]
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
    elif page == "Отчёт":
        page_report(df, cfg)


if __name__ == "__main__":
    main()
