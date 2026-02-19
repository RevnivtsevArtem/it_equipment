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
    return pd.read_csv("data/sample_tickets.csv")


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
        st.download_button("Скачать CSV", data=buf.getvalue(),
                           file_name="current_dataset.csv", mime="text/csv")



def _list_saved_models(models_dir: str = "models") -> list[str]:
    """Возвращает список сохранённых .pkl моделей (относительные пути)."""
    if not os.path.isdir(models_dir):
        return []
    paths = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))
    # приводим к unix-пути для красивого отображения
    return [p.replace('\\', '/') for p in paths]


@st.cache_resource(show_spinner=False)
def _cached_load_model(model_path: str):
    """Кэширует загрузку моделей, чтобы не читать .pkl каждый раз."""
    return load_model(model_path)


def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    """Вкладка прогнозирования вероятности замены устройства по сохранённым моделям."""
    st.subheader("Прогноз необходимости замены устройства")

    target_col = cfg["default_target_column"]
    feature_cols = cfg["categorical_columns"] + cfg["numeric_columns"]

    # 1) Выбор записи из датасета — без ручного ввода параметров
    if any(col not in df.columns for col in feature_cols):
        missing = [c for c in feature_cols if c not in df.columns]
        st.error("В данных отсутствуют необходимые признаки: " + ", ".join(missing))
        return

    if len(df) == 0:
        st.warning("Датасет пуст — нечего прогнозировать.")
        return

    st.caption("Выберите строку из загруженного датасета — параметры подставятся автоматически.")
    row_idx = st.selectbox("Строка датасета", options=list(df.index), index=0)
    df_input = df.loc[[row_idx], feature_cols].copy()

    st.write("Параметры выбранного устройства (из датасета):")
    st.dataframe(df_input)

    # 2) Порог классификации
    threshold = st.slider(
        "Порог принятия решения (threshold)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Если вероятность >= порога, устройство считается требующим обновления.",
    )

    st.info(
        f"Интерпретация: при threshold = {threshold:.2f} модель помечает устройство как "
        f"«нуждается в обновлении», если вероятность замены ≥ {threshold:.2f}."
    )

    # 3) Выбор и загрузка сохранённых моделей
    model_paths = _list_saved_models("models")
    if not model_paths:
        st.warning(
            "В папке `models/` не найдено ни одной сохранённой модели (.pkl). "
            "Сначала обучите модели во вкладке «Обучение» и сохраните их."
        )
        return

    st.subheader("Вероятности по всем сохранённым моделям")

    results = []
    for model_path in model_paths:
        try:
            model = _cached_load_model(model_path)
            # predict_needs_upgrade возвращает (labels, proba)
            _, proba = predict_needs_upgrade(model=model, df_inputs=df_input, feature_cols=feature_cols)
            p = float(proba[0])
            label = int(p >= threshold)
            results.append(
                {
                    "model_path": model_path,
                    "probability": p,
                    "decision": "Нуждается в обновлении" if label == 1 else "Пока не требуется",
                }
            )
        except Exception as e:
            results.append(
                {
                    "model_path": model_path,
                    "probability": float("nan"),
                    "decision": f"Ошибка: {e}",
                }
            )

    res_df = pd.DataFrame(results)
    # сортируем по вероятности, чтобы сразу было видно лидеров
    if "probability" in res_df.columns:
        res_df = res_df.sort_values(by="probability", ascending=False, na_position="last")

    st.dataframe(res_df, use_container_width=True)

    # 4) Красивые индикаторы: прогресс-бары для топ-N (или всех, если мало)
    st.subheader("Индикаторы вероятности")
    max_show = min(10, len(res_df))
    st.caption(f"Показаны индикаторы для первых {max_show} моделей из списка.")
    for _, r in res_df.head(max_show).iterrows():
        st.write(f"**{r['model_path']}** — {r['decision']}")
        if pd.isna(r["probability"]):
            st.error("Не удалось получить вероятность для этой модели.")
            continue
        st.progress(int(round(r["probability"] * 100)))
        st.caption(f"Вероятность замены: **{r['probability']:.3f}** (т.е. {r['probability']*100:.1f}%)")


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
        "Имя файла модели", value=f"{model_name.lower()}_full.pkl")
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
        st.success(f"Модель сохранена в {model_path}")
        st.json(metrics)


def page_model_comparison(df: pd.DataFrame, cfg: dict) -> None:
    from sklearn.model_selection import train_test_split

    st.subheader("Сравнение моделей")
    target_col = cfg["default_target_column"]

    df_split = df.dropna(subset=[target_col]).copy()
    df_split[target_col] = pd.to_numeric(
        df_split[target_col], errors="coerce").dropna().astype(int)

    df_train, df_test = train_test_split(df_split, test_size=0.2, random_state=42)

    all_models = [
        "LogisticRegression", "KNN", "RandomForest",
        "GradientBoosting", "ExtraTrees", "MLPClassifier"
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


def page_eda(df: pd.DataFrame) -> None:
    st.subheader("Разведочный анализ данных (EDA)")
    st.write("Всего записей:", len(df))
    st.pyplot(plot_ticket_counts_by_department(df))
    st.pyplot(plot_ticket_counts_by_device_type(df))
    st.pyplot(plot_device_age_hist(df))
    st.pyplot(plot_tickets_last_6_months_hist(df))


# ================= НОВАЯ СТРАНИЦА ПРОГНОЗА =================

def page_prediction(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Прогноз вероятности замены устройства")

    models_folder = "models"
    if not os.path.exists(models_folder):
        st.warning("Папка models не найдена.")
        return

    model_files = [f for f in os.listdir(models_folder) if f.endswith(".pkl")]

    if not model_files:
        st.warning("В папке models нет обученных моделей.")
        return

    threshold = st.slider("Порог принятия решения", 0.0, 1.0, 0.5, 0.01)

    input_data = {}

    for col in cfg["categorical_columns"]:
        input_data[col] = st.selectbox(
            col, sorted(df[col].dropna().unique()))

    for col in cfg["numeric_columns"]:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("Рассчитать вероятность по всем моделям"):
        input_df = pd.DataFrame([input_data])

        for model_file in model_files:
            model_path = os.path.join(models_folder, model_file)
            model = load_model(model_path)

            labels, proba = predict_needs_upgrade(
                model,
                input_df,
                cfg["categorical_columns"] + cfg["numeric_columns"],
            )

            st.write(f"### {model_file}")
            st.progress(float(proba[0]))
            st.write(f"Вероятность замены: {proba[0]*100:.2f}%")
            decision = "ТРЕБУЕТСЯ замена" if proba[0] >= threshold else "Замена не требуется"
            st.write(f"Решение при пороге {threshold}: {decision}")
            st.markdown("---")


def main() -> None:
    st.set_page_config(
        page_title="Прогноз обновления вычислительной техники",
        layout="wide"
    )

    st.title(
        "Интеллектуальная система прогнозирования потребностей "
        "в обновлении вычислительной техники"
    )

    cfg = load_config()

    st.sidebar.header("Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV с обращениями", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = _load_default_data()
        st.info("Используется демонстрационный датасет data/sample_tickets.csv")

    page = st.sidebar.radio(
        "Раздел приложения",
        [
            "Обзор",
            "Данные",
            "Обучение",
            "Сравнение моделей",
            "EDA",
            "Прогноз устройства"
        ],
    )

    if page == "Обзор":
        page_overview()
    elif page == "Данные":
        page_data(df, cfg)
    elif page == "Прогноз":
        page_prediction(df, cfg)
    elif page == "Обучение":
        page_training(df, cfg)
    elif page == "Сравнение моделей":
        page_model_comparison(df, cfg)
    elif page == "EDA":
        page_eda(df)
    elif page == "Прогноз устройства":
        page_prediction(df, cfg)


if __name__ == "__main__":
    main()