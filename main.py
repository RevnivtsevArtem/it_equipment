# -*- coding: utf-8 -*-
"""
Главное web-приложение Streamlit.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
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


def _load_default_data() -> pd.DataFrame:
    """Загружает демонстрационный датасет."""
    return pd.read_csv("data/sample_tickets.csv")


def _normalize_to_list(value) -> list[str]:
    """
    Приводит значение из config к списку.
    - list -> list
    - str  -> split по запятой
    - None/другое -> []
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _required_columns(cfg: dict) -> list[str]:
    """Список обязательных столбцов для работы приложения."""
    cols: list[str] = []
    cols.extend(_normalize_to_list(cfg.get("categorical_columns")))
    cols.extend(_normalize_to_list(cfg.get("numeric_columns")))

    target = cfg.get("default_target_column")
    if isinstance(target, str) and target.strip():
        cols.append(target.strip())

    return list(dict.fromkeys(cols))


def _try_read_csv(uploaded_file, sep: str) -> pd.DataFrame:
    """
    Читает CSV из uploaded_file устойчиво к BOM/кодировкам и к неверному разделителю.
    Если выбранный sep даёт 1 столбец, пытается альтернативные (',', ';', '\\t').
    """
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        df = pd.read_csv(uploaded_file, sep=sep, encoding="utf-8-sig")
    except TypeError:
        df = pd.read_csv(uploaded_file, sep=sep)

    if df.shape[1] <= 1:
        for candidate in [",", ";", "\t"]:
            if candidate == sep:
                continue
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

            try:
                df2 = pd.read_csv(uploaded_file, sep=candidate, encoding="utf-8-sig")
            except TypeError:
                df2 = pd.read_csv(uploaded_file, sep=candidate)

            if df2.shape[1] > 1:
                return df2

    return df


def _validate_uploaded_df(df: pd.DataFrame, cfg: dict) -> tuple[bool, list[str]]:
    """Проверяет, что в df есть все нужные колонки."""
    required = _required_columns(cfg)
    missing = [c for c in required if c not in df.columns]
    ok = len(missing) == 0
    return ok, missing


def page_overview() -> None:
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


def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз необходимости замены устройства")

    target_col = cfg["default_target_column"]
    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует в данных.")
        return

    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    if df_clean.empty:
        st.error("Нет корректных данных для прогнозирования.")
        return

    selected_index = st.selectbox("Выберите устройство (строку датасета)", df_clean.index)

    st.write("Параметры выбранного устройства:")
    st.dataframe(df_clean.loc[[selected_index]])

    if st.button("Спрогнозировать по всем моделям"):
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

            results.append({"model": model_name, "probability": float(proba[0])})

        if not results:
            st.error("Нет сохранённых моделей. Сначала обучите модели.")
            return

        results_df = pd.DataFrame(results)

        st.subheader("Вероятности по каждой модели")
        st.dataframe(results_df)

        avg_proba = results_df["probability"].mean()

        st.subheader("Ансамблевый прогноз (среднее значение)")
        st.progress(avg_proba)
        st.markdown(f"### Средняя вероятность: {avg_proba:.2%}")

        if avg_proba >= 0.7:
            st.error("Высокая вероятность необходимости замены устройства")
        elif avg_proba >= 0.5:
            st.warning("Средняя вероятность необходимости замены устройства")
        else:
            st.success("Низкая вероятность необходимости необходимости замены устройства")


def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    import io as _io
    from sklearn.model_selection import train_test_split

    st.subheader("Сравнение моделей")

    target_col = cfg["default_target_column"]

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует в данных.")
        return

    test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    df_split = df.dropna(subset=[target_col]).copy()
    df_split[target_col] = pd.to_numeric(df_split[target_col], errors="coerce")
    df_split = df_split.dropna(subset=[target_col])
    df_split[target_col] = df_split[target_col].astype(int)

    if df_split[target_col].nunique() < 2:
        df_train, df_test = train_test_split(df_split, test_size=test_size, random_state=random_state)
    else:
        df_train, df_test = train_test_split(
            df_split,
            test_size=test_size,
            random_state=random_state,
            stratify=df_split[target_col],
        )

    all_models = [
        "LogisticRegression",
        "KNN",
        "RandomForest",
        "GradientBoosting",
        "ExtraTrees",
        "MLPClassifier",
    ]

    if st.button("Обучить и сравнить модели"):
        report_df = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=target_col,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=all_models,
        )

        st.dataframe(report_df)

        for _, row in report_df.iterrows():
            st.write(f"Модель {row['model']}: accuracy = {row.get('accuracy', '—')}")

        buf = _io.StringIO()
        report_df.to_csv(buf, index=False)
        st.download_button(
            "Скачать отчёт по всем моделям (CSV)",
            data=buf.getvalue(),
            file_name="models_comparison_report.csv",
            mime="text/csv",
        )


def page_eda(df: pd.DataFrame) -> None:
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
    import io as _io
    from sklearn.model_selection import train_test_split

    st.subheader("Демонстрация моделей и итоговый отчёт")

    target_col = cfg["default_target_column"]

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует в данных.")
        return

    test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    df_split = df.dropna(subset=[target_col]).copy()
    df_split[target_col] = pd.to_numeric(df_split[target_col], errors="coerce")
    df_split = df_split.dropna(subset=[target_col])
    df_split[target_col] = df_split[target_col].astype(int)

    if df_split[target_col].nunique() < 2:
        df_train, df_test = train_test_split(df_split, test_size=test_size, random_state=random_state)
    else:
        df_train, df_test = train_test_split(
            df_split,
            test_size=test_size,
            random_state=random_state,
            stratify=df_split[target_col],
        )

    all_models = [
        "LogisticRegression",
        "KNN",
        "RandomForest",
        "GradientBoosting",
        "ExtraTrees",
        "MLPClassifier",
    ]

    if st.button("Сформировать отчёт по всем моделям"):
        report_df = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=target_col,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=all_models,
        )

        st.subheader("Итоговый отчёт по всем моделям")
        st.dataframe(report_df)

        for _, row in report_df.iterrows():
            st.write(f"Модель {row['model']}: accuracy = {row.get('accuracy', '—')}")

        buf = _io.StringIO()
        report_df.to_csv(buf, index=False)
        st.download_button(
            "Скачать отчёт по всем моделям (CSV)",
            data=buf.getvalue(),
            file_name="all_models_report.csv",
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

    # ==========================================================
    # 1) НАВИГАЦИЯ
    # ==========================================================
    st.sidebar.header("Разделы приложения")
    page = st.sidebar.radio(
        "Раздел приложения",
        ["Обзор", "Данные", "Обучение", "Сравнение моделей", "Прогноз устройства", "EDA", "Отчёт"],
    )
    st.sidebar.divider()

    # ==========================================================
    # 2) ЗАГРУЗКА ДАННЫХ  (ВАЖНО: ВНУТРИ main()!)
    # ==========================================================
    st.sidebar.header("Загрузка данных")

    if "current_df" not in st.session_state:
        st.session_state.current_df = _load_default_data()
        st.session_state.dataset_source = "demo"

    with st.sidebar.expander("Требования к датасету", expanded=False):
        req = _required_columns(cfg)
        st.write("Обязательные столбцы:")
        if req:
            st.code("\n".join(req), language="text")
        else:
            st.warning("Не удалось получить список обязательных столбцов.")

    sep = st.sidebar.selectbox("Разделитель CSV", [",", ";", "\t"])

    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV с обращениями",
        type=["csv"]
    )

    reset_demo = st.sidebar.button("Сброс на демо")

    if reset_demo:
        st.session_state.current_df = _load_default_data()
        st.session_state.dataset_source = "demo"
        st.sidebar.success("Загружен демонстрационный датасет.")

    # ВАЖНО: без кнопки "Применить", чтобы не ломать rerun
    if uploaded_file is not None:
        try:
            df_uploaded = _try_read_csv(uploaded_file, sep=sep)
            ok, missing = _validate_uploaded_df(df_uploaded, cfg)

            if not ok:
                st.sidebar.error("CSV загружен, но структура не подходит.")
                st.sidebar.code("\n".join(missing), language="text")
            else:
                st.session_state.current_df = df_uploaded
                st.session_state.dataset_source = "user"
                st.sidebar.success("Пользовательский датасет загружен.")
        except Exception as e:
            st.sidebar.error(f"Ошибка чтения CSV: {e}")

    df = st.session_state.current_df

    if st.session_state.dataset_source == "demo":
        st.info("Используется демонстрационный датасет.")
    else:
        st.success("Используется пользовательский датасет.")

    # ==========================================================
    # 3) РЕНДЕР СТРАНИЦ (ДОЛЖЕН БЫТЬ ВСЕГДА, НЕ ТОЛЬКО В else!)
    # ==========================================================
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
