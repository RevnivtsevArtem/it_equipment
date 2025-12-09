import pandas as pd

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # device_age_years: map categories to numeric
    age_map = {"low": 1, "medium": 3, "high": 5}
    if "device_age_years" in df.columns:
        df["device_age_years"] = df["device_age_years"].map(age_map).fillna(3)

    # tickets_last_6_months: use text length as numeric proxy
    if "tickets_last_6_months" in df.columns:
        df["tickets_last_6_months"] = (
            df["tickets_last_6_months"]
            .astype(str)
            .apply(lambda x: len(x.strip()))
        )

    if "problem_description" in df.columns:
        df["problem_description"] = df["problem_description"].astype(str)

    return df
