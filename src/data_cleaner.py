import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------- Очистка целевой переменной ----------
    if "needs_upgrade" in df.columns:
        # удаляем записи без целевого значения
        df = df.dropna(subset=["needs_upgrade"])

        # приводим к целочисленному бинарному виду
        df["needs_upgrade"] = df["needs_upgrade"].astype(int)

    # ---------- Обработка возраста устройства ----------
    # поддержка как числовых, так и категориальных значений
    age_map = {
        "low": 1,
        "medium": 3,
        "high": 5
    }

    if "device_age_years" in df.columns:
        df["device_age_years"] = (
            df["device_age_years"]
            .replace(age_map)
            .astype(float)
            .fillna(df["device_age_years"].median())
        )

    # ---------- Интенсивность обращений ----------
    # если значение нечисловое — приводим к числу
    if "tickets_last_6_months" in df.columns:
        df["tickets_last_6_months"] = pd.to_numeric(
            df["tickets_last_6_months"],
            errors="coerce"
        ).fillna(0)

    # ---------- Текстовое поле ----------
    if "problem_description" in df.columns:
        df["problem_description"] = (
            df["problem_description"]
            .astype(str)
            .str.strip()
        )

    return df
