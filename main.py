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
    evaluate_all_models,
)

from src.eda import (
    plot_ticket_counts_by_department,
    plot_ticket_counts_by_device_type,
    plot_device_age_hist,
    plot_tickets_last_6_months_hist,
)

# ==========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================================

def _load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/sample_tickets.csv")


def _normalize_to_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _required_columns(cfg: dict) -> list[str]:
    cols = []
    cols.extend(_normalize_to_list(cfg.get("categorical_columns")))
    cols.extend(_normalize_to_list(cfg.get("numeric_columns")))
    target = cfg.get("default_target_column")
    if isinstance(target, str):
        cols.append(target)
    return list(dict.fromkeys(cols))


def _try_read_csv(uploaded_file, sep: str) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, sep=sep, encoding="utf-8-sig")
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=sep)

    if df.shape[1] <= 1:
        for candidate in [",", ";", "\t"]:
            if candidate == sep:
                continue
            uploaded_file.seek(0)
            try:
                df2 = pd.read_csv(uploaded_file, sep=candidate, encoding="utf-8-sig")
            except Exception:
                df2 = pd.read_csv(uploaded_file, sep=candidate)
            if df2.shape[1] > 1:
                return df2

    return df


def _validate_uploaded_df(df: pd.DataFrame, cfg: dict):
    required = _required_columns(cfg)
    missing = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing


# ==========================================================
# СТРАНИЦЫ
# ==========================================================

def page_overview():
    st.subheader("Общая информация о системе")
    st.write(
        """
        Интеллектуальная система прогнозирования потребностей
        в обновлении вычислительной техники на основе данных службы поддержки.
        """
    )


def page_data(df, cfg):
    st.subheader("Работа с данными")
    st.write("Размер датасета:", df.shape)
    st.dataframe(df.head(20))

    if st.checkbox("Показать статистику"):
        st.write(df.describe(include="all"))


def page_training(df, cfg):
    st.subheader("Обучение модели")

    target = cfg["default_target_column"]

    if target not in df.columns:
        st.error("В датасете отсутствует целевой столбец.")
        return

    model_name = st.selectbox(
        "Выберите модель",
        ["LogisticRegression", "KNN", "RandomForest",
         "GradientBoosting", "ExtraTrees", "MLPClassifier"]
    )

    if st.button("Обучить и сохранить модель"):
        pipeline, metrics = fit_on_full_and_save(
            df=df,
            target_column=target,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_name=model_name,
            model_path=f"models/{model_name.lower()}_full.pkl"
        )
        st.success("Модель успешно обучена и сохранена.")
        st.json(metrics)


def page_prediction(df, cfg):
    st.subheader("Прогноз устройства")

    target = cfg["default_target_column"]
    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    if target not in df.columns:
        st.error("Нет целевого столбца.")
        return

    df_clean = df.dropna(subset=feature_cols)

    if df_clean.empty:
        st.error("Нет данных для прогноза.")
        return

    idx = st.selectbox("Выберите запись", df_clean.index)
    st.dataframe(df_clean.loc[[idx]])

    if st.button("Спрогнозировать"):
        results = []

        for model_name in [
            "LogisticRegression", "KNN", "RandomForest",
            "GradientBoosting", "ExtraTrees", "MLPClassifier"
        ]:
            path = f"models/{model_name.lower()}_full.pkl"
            if not os.path.exists(path):
                continue
            model = load_model(path)
            _, proba = predict_needs_upgrade(
                model=model,
                df_inputs=df_clean.loc[[idx]],
                feature_cols=feature_cols
            )
            results.append((model_name, float(proba[0])))

        if not results:
            st.error("Нет обученных моделей.")
            return

        for model, p in results:
            st.write(f"{model}: {p:.2%}")


def page_model_comparison(df, cfg):
    st.subheader("Сравнение моделей")

    from sklearn.model_selection import train_test_split

    target = cfg["default_target_column"]
    df_clean = df.dropna(subset=[target])
    train, test = train_test_split(df_clean, test_size=0.2, random_state=42)

    if st.button("Обучить и сравнить"):
        report = evaluate_all_models(
            df_train=train,
            df_test=test,
            target_column=target,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=[
                "LogisticRegression", "KNN", "RandomForest",
                "GradientBoosting", "ExtraTrees", "MLPClassifier"
            ]
        )
        st.dataframe(report)


def page_eda(df):
    st.subheader("EDA")
    st.pyplot(plot_ticket_counts_by_department(df))
    st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


def page_report(df, cfg):
    st.subheader("Итоговый отчёт")

    if st.button("Сформировать отчёт"):
        from sklearn.model_selection import train_test_split
        target = cfg["default_target_column"]
        df_clean = df.dropna(subset=[target])
        train, test = train_test_split(df_clean, test_size=0.2, random_state=42)

        report = evaluate_all_models(
            df_train=train,
            df_test=test,
            target_column=target,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=[
                "LogisticRegression", "KNN", "RandomForest",
                "GradientBoosting", "ExtraTrees", "MLPClassifier"
            ]
        )
        st.dataframe(report)


# ==========================================================
# MAIN
# ==========================================================

def main():
    st.set_page_config(layout="wide")
    st.title("Интеллектуальная система прогнозирования")

    cfg = load_config()

    # -------- Навигация --------
    st.sidebar.header("Разделы приложения")
    page = st.sidebar.radio(
        "Выберите раздел",
        ["Обзор", "Данные", "Обучение",
         "Сравнение моделей", "Прогноз устройства",
         "EDA", "Отчёт"]
    )

    # -------- Загрузка данных --------
    st.sidebar.header("Загрузка данных")

    if "current_df" not in st.session_state:
        st.session_state.current_df = _load_default_data()
        st.session_state.dataset_source = "demo"

    with st.sidebar.expander("Требования к датасету"):
        for col in _required_columns(cfg):
            st.markdown(f"- `{col}`")

    sep = st.sidebar.selectbox("Разделитель CSV", [",", ";", "\t"])

    uploaded = st.sidebar.file_uploader("Загрузите CSV", type=["csv"])

    if st.sidebar.button("Применить файл"):
        if uploaded is None:
            st.sidebar.error("Выберите файл.")
        else:
            df_up = _try_read_csv(uploaded, sep)
            ok, missing = _validate_uploaded_df(df_up, cfg)
            if not ok:
                st.sidebar.error("Структура не соответствует требованиям.")
                st.sidebar.code("\n".join(missing))
            else:
                st.session_state.current_df = df_up
                st.session_state.dataset_source = "user"
                st.sidebar.success("Файл успешно применён.")

    if st.sidebar.button("Сброс на демо"):
        st.session_state.current_df = _load_default_data()
        st.session_state.dataset_source = "demo"

    df = st.session_state.current_df

    # -------- Рендер страниц --------
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
