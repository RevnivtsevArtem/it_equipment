import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка и приведение датасета к виду,
    пригодному для обучения моделей машинного обучения.
    """
    df = df.copy()

    # --- Целевая переменная ---
    if "needs_upgrade" in df.columns:
        # удаляем строки без целевой метки
        df = df[df["needs_upgrade"].notna()]
        # приводим к бинарному целочисленному типу
        df["needs_upgrade"] = df["needs_upgrade"].astype(int)

    # --- Возраст устройства ---
    if "device_age_years" in df.columns:
        age_map = {"low": 1, "medium": 3, "high": 5}
        df["device_age_years"] = (
            df["device_age_years"]
            .replace(age_map)
            .astype(float)
            .fillna(df["device_age_years"].median())
        )

    # --- Количество обращений ---
    if "tickets_last_6_months" in df.columns:
        df["tickets_last_6_months"] = (
            df["tickets_last_6_months"]
            .astype(str)
            .apply(lambda x: len(x.strip()))
        )

    # --- Текстовое описание ---
    if "problem_description" in df.columns:
        df["problem_description"] = df["problem_description"].astype(str)

    # --- Категориальные признаки ---
    categorical_cols = [
        "user_department",
        "device_type",
        "priority",
        "os",
        "location",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")

    return df
