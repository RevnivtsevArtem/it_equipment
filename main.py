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

# ВАЖНО: первая Streamlit-команда
st.set_page_config(
    page_title="Прогноз обновления вычислительной техники",
    layout="wide",
    initial_sidebar_state="expanded",
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
    st.dataframe(df.head(20), use_container_width=True)

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
            _, metrics = fit_on_full_and_save(
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
    """
    Вкладка прогнозирования:
    - выбор одной модели или ВСЕХ моделей из models/*.pkl
    - категории через selectbox из датасета
    - вероятности сразу по всем моделям (если выбрано "ВСЕ модели")
    - threshold (slider) + интерпретация решения
    """
    st.subheader("Прогноз необходимости замены устройства")

    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    # 1) Модели
    model_paths = sorted(glob.glob("models/*.pkl"))
    if not model_paths:
        st.warning("В папке `models/` не найдено ни одной модели (.pkl). Сначала обучите модель во вкладке «Обучение».")
        return

    model_names = [os.path.basename(p) for p in model_paths]
    selected = st.selectbox("Модель для прогноза", ["ВСЕ модели"] + model_names)

    # 2) Параметры устройства
    st.markdown("### Параметры устройства")
    input_data: dict[str, object] = {}

    for col in cfg["categorical_columns"]:
        values = sorted(df[col].dropna().astype(str).unique())
        if not values:
            st.error(f"В столбце `{col}` нет значений для выбора.")
            return
        input_data[col] = st.selectbox(col, values)

    for col in cfg["numeric_columns"]:
        default_val = float(pd.to_numeric(df[col], errors="coerce").mean())
        if pd.isna(default_val):
            default_val = 0.0
        input_data[col] = st.number_input(col, value=default_val)

    input_df = pd.DataFrame([input_data])

    # 3) Threshold
    threshold = st.slider(
        "Порог принятия решения (threshold)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Если вероятность >= порога, устройство считается требующим обновления.",
    )
    st.info(f"Интерпретация: при threshold = {threshold:.2f} решение «ТРЕБУЕТСЯ обновление» принимается при вероятности ≥ {threshold:.2f}.")

    # 4) Расчёт
    if st.button("Спрогнозировать"):
        paths_to_use = model_paths if selected == "ВСЕ модели" else [p for p in model_paths if os.path.basename(p) == selected]
        results = []
        for pth in paths_to_use:
            name = os.path.basename(pth)
            try:
                model = load_model(pth)
                _, proba = predict_needs_upgrade(
                    model=model,
                    df_inputs=input_df,
                    feature_cols=feature_cols,
                )
                prob = float(proba[0])
                decision = "ТРЕБУЕТСЯ обновление" if prob >= threshold else "Пока не требуется"
                results.append({"Модель": name, "Вероятность": prob, "Решение": decision})
            except Exception as e:
                results.append({"Модель": name, "Вероятность": float("nan"), "Решение": f"Ошибка: {e}"})

        res_df = pd.DataFrame(results).sort_values(by="Вероятность", ascending=False, na_position="last")
        st.markdown("## Результаты прогнозирования")
        st.dataframe(res_df, use_container_width=True)

        st.markdown("### Индикаторы вероятности")
        for _, row in res_df.iterrows():
            st.write(f"**{row['Модель']}** — {row['Решение']}")
            if pd.isna(row["Вероятность"]):
                st.error("Не удалось получить вероятность для модели.")
                continue
            st.progress(int(round(float(row["Вероятность"]) * 100)))
            st.caption(f"Вероятность: {float(row['Вероятность'])*100:.2f}%")


def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    """
    Сравнение моделей (как в твоём коде):
    - train/test split
    - stratify при наличии 2 классов
    - evaluate_all_models(df_train, df_test, ...)
    """
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

    if len(df_split) < 5:
        st.warning("Слишком мало данных для сравнения моделей.")
        return

    if df_split[target_col].nunique() < 2:
        df_train, df_test = train_test_split(df_split, test_size=test_size, random_state=int(random_state))
    else:
        df_train, df_test = train_test_split(
            df_split,
            test_size=test_size,
            random_state=int(random_state),
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

        st.dataframe(report_df, use_container_width=True)

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


def _safe_show_model_eval(eval_out: object) -> None:
    """
    Пытается отрисовать Confusion Matrix / ROC / PR + таблицу метрик.
    Работает с разными форматами возврата fit_model_and_evaluate.
    """
    # Возможные варианты: (pipeline, metrics, y_true, y_pred, y_proba) или dict.
    pipeline = None
    metrics = None
    y_true = None
    y_pred = None
    y_proba = None

    if isinstance(eval_out, dict):
        metrics = eval_out.get("metrics") or eval_out
        y_true = eval_out.get("y_true")
        y_pred = eval_out.get("y_pred")
        y_proba = eval_out.get("y_proba")
    elif isinstance(eval_out, (list, tuple)):
        # самый частый ожидаемый формат
        if len(eval_out) >= 2:
            pipeline = eval_out[0]
            metrics = eval_out[1]
        if len(eval_out) >= 3:
            y_true = eval_out[2]
        if len(eval_out) >= 4:
            y_pred = eval_out[3]
        if len(eval_out) >= 5:
            y_proba = eval_out[4]

    if metrics is not None:
        try:
            st.markdown("### Метрики качества")
            st.markdown(metrics_to_markdown_table(metrics))
        except Exception:
            st.json(metrics)

    # Графики
    cols = st.columns(2)
    try:
        if y_true is not None and y_pred is not None:
            with cols[0]:
                st.pyplot(plot_confusion_matrix(y_true, y_pred))
    except Exception as e:
        st.warning(f"Не удалось построить confusion matrix: {e}")

    try:
        if y_true is not None and y_proba is not None:
            with cols[1]:
                st.pyplot(plot_roc_curve(y_true, y_proba))
    except Exception as e:
        st.warning(f"Не удалось построить ROC: {e}")

    try:
        if y_true is not None and y_proba is not None:
            st.pyplot(plot_pr_curve(y_true, y_proba))
    except Exception as e:
        st.warning(f"Не удалось построить PR: {e}")


def page_report(df: pd.DataFrame, cfg: dict) -> None:
    """
    Итоговый отчёт (с сохранением идеи твоей страницы):
    - формирование отчёта по всем моделям (evaluate_all_models)
    - детальная демонстрация выбранной модели (fit_model_and_evaluate)
    - Confusion Matrix / ROC / PR + таблица метрик
    """
    from sklearn.model_selection import train_test_split

    st.subheader("Демонстрация моделей и итоговый отчёт")

    target_col = cfg["default_target_column"]

    if target_col not in df.columns:
        st.error(f"Целевой столбец `{target_col}` отсутствует в данных.")
        return

    test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2, 0.05, key="rep_test_size")
    random_state = st.number_input("Random state", value=42, step=1, key="rep_rs")

    df_split = df.dropna(subset=[target_col]).copy()
    df_split[target_col] = pd.to_numeric(df_split[target_col], errors="coerce")
    df_split = df_split.dropna(subset=[target_col])
    df_split[target_col] = df_split[target_col].astype(int)

    if len(df_split) < 10:
        st.warning("Слишком мало данных для отчёта. Загрузите полный датасет.")
        return

    if df_split[target_col].nunique() < 2:
        df_train, df_test = train_test_split(df_split, test_size=test_size, random_state=int(random_state))
    else:
        df_train, df_test = train_test_split(
            df_split,
            test_size=test_size,
            random_state=int(random_state),
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

    st.markdown("### 1) Сводный отчёт по всем моделям")
    if st.button("Сформировать отчёт по всем моделям"):
        report_df = evaluate_all_models(
            df_train=df_train,
            df_test=df_test,
            target_column=target_col,
            categorical_cols=cfg["categorical_columns"],
            numeric_cols=cfg["numeric_columns"],
            model_names=all_models,
        )
        st.dataframe(report_df, use_container_width=True)

        buf = io.StringIO()
        report_df.to_csv(buf, index=False)
        st.download_button(
            "Скачать отчёт (CSV)",
            data=buf.getvalue(),
            file_name="all_models_report.csv",
            mime="text/csv",
        )

    st.markdown("### 2) Детальная оценка выбранной модели")
    model_name = st.selectbox("Модель для детальной оценки", all_models, key="rep_model")

    if st.button("Построить графики и метрики выбранной модели"):
        with st.spinner("Выполняется обучение и оценка..."):
            try:
                # Пытаемся угадать сигнатуру (позиционно, чтобы не словить unexpected keyword)
                eval_out = fit_model_and_evaluate(
                    df_train,
                    df_test,
                    target_col,
                    cfg["categorical_columns"],
                    cfg["numeric_columns"],
                    model_name,
                )
                _safe_show_model_eval(eval_out)
            except TypeError:
                # альтернативная сигнатура с keyword-аргументами (если такая у тебя)
                eval_out = fit_model_and_evaluate(
                    df_train=df_train,
                    df_test=df_test,
                    target_column=target_col,
                    categorical_cols=cfg["categorical_columns"],
                    numeric_cols=cfg["numeric_columns"],
                    model_name=model_name,
                )
                _safe_show_model_eval(eval_out)
            except Exception as e:
                st.error(f"Не удалось выполнить оценку модели: {e}")


def main() -> None:
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

    # Sidebar (всегда рендерим меню — никаких return до radio)
    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV с обращениями", type=["csv"])

    # Датасет
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Данные успешно загружены.")
    else:
        df = _load_default_data()
        st.info("Используется демонстрационный датасет `data/sample_tickets.csv`.")

    # Меню
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
