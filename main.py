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


def _list_saved_models(models_dir: str = "models") -> list[str]:
    """Возвращает список сохранённых моделей (*.pkl) из папки models/."""
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(models_dir, "*.pkl"))])


def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    """
    Вкладка прогнозирования вероятности замены устройства.
    Требования:
    - без ручного ввода пути к модели: выбор из выпадающего списка всех models/*.pkl
    - категориальные признаки: auto-подстановка значений через selectbox
    - отображение вероятностей по всем моделям сразу
    - интерпретация порога: настраиваемый threshold
    """
    st.subheader("Прогноз необходимости замены устройства")

    models_dir = "models"
    model_files = _list_saved_models(models_dir)

    if not model_files:
        st.warning("В папке models не найдено сохранённых моделей (*.pkl). Сначала обучите и сохраните модель.")
        return

    threshold = st.slider("Порог принятия решения (threshold)", 0.0, 1.0, 0.5, 0.01)

    st.markdown("### Параметры устройства")

    input_data: dict = {}

    # Категориальные: selectbox по уникальным значениям из текущего df
    for col in cfg["categorical_columns"]:
        if col in df.columns:
            values = df[col].dropna().unique().tolist()
            values_sorted = sorted(values, key=lambda x: str(x))
            if len(values_sorted) == 0:
                # если в колонке нет значений — даём безопасный ввод
                input_data[col] = st.text_input(col, value="")
            else:
                input_data[col] = st.selectbox(col, values_sorted)
        else:
            # если колонка отсутствует в df — безопасный ввод, чтобы приложение не падало
            input_data[col] = st.text_input(col, value="")

    # Числовые: number_input с дефолтом = среднее (если есть), иначе 0.0
    for col in cfg["numeric_columns"]:
        if col in df.columns:
            try:
                default_val = float(pd.to_numeric(df[col], errors="coerce").dropna().mean())
                if pd.isna(default_val):
                    default_val = 0.0
            except Exception:
                default_val = 0.0
        else:
            default_val = 0.0
        input_data[col] = st.number_input(col, value=float(default_val))

    if st.button("Рассчитать вероятности по всем моделям"):
        df_input = pd.DataFrame([input_data])
        feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

        st.markdown("## Результаты по моделям")

        # Табличный блок: быстрее сравнивать
        results_rows = []

        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            try:
                model = load_model(model_path)
                _, proba = predict_needs_upgrade(
                    model=model,
                    df_inputs=df_input,
                    feature_cols=feature_cols,
                )
                probability = float(proba[0])
            except Exception as e:
                # не валим весь раздел из-за одной модели
                probability = None
                results_rows.append({"model": model_file, "probability": None, "decision": f"Ошибка: {e}"})
                continue

            decision = "ТРЕБУЕТСЯ замена" if probability >= threshold else "Замена не требуется"
            results_rows.append(
                {"model": model_file, "probability": probability, "decision": decision}
            )

        # Показываем таблицу (без падения, если есть ошибки)
        result_df = pd.DataFrame(results_rows)
        if "probability" in result_df.columns:
            st.dataframe(
                result_df.assign(
                    probability_percent=result_df["probability"].apply(
                        lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else "—"
                    )
                ).drop(columns=["probability"]),
                use_container_width=True,
            )
        else:
            st.dataframe(result_df, use_container_width=True)

        st.markdown("## Индикаторы вероятности")
        for row in results_rows:
            st.markdown(f"### {row['model']}")
            if isinstance(row.get("probability"), (int, float)):
                p = float(row["probability"])
                st.progress(p)
                st.markdown(f"**Вероятность необходимости замены:** {p:.2%}")
                if p >= threshold:
                    st.error(f"{row['decision']} (p ≥ {threshold:.2f})")
                else:
                    st.success(f"{row['decision']} (p < {threshold:.2f})")
            else:
                st.warning(row.get("decision", "Не удалось рассчитать вероятность."))


def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    import io
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
        df_train, df_test = train_test_split(
            df_split,
            test_size=test_size,
            random_state=random_state,
        )
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

        buf = io.StringIO()
        report_df.to_csv(buf, index=False)
        st.download_button(
            "Скачать отчёт по всем моделям (CSV)",
            data=buf.getvalue(),
            file_name="models_comparison_report.csv",
            mime="text/csv",
        )


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
    import io
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
        df_train, df_test = train_test_split(
            df_split,
            test_size=test_size,
            random_state=random_state,
        )
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

        buf = io.StringIO()
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
        ["Обзор", "Данные", "Обучение", "Сравнение моделей", "Прогноз устройства", "EDA", "Отчёт"],
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
