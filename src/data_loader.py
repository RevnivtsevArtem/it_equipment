# -*- coding: utf-8 -*-
"""
Модуль загрузки и первичной проверки данных.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники
на основе обращений в службу технической поддержки (на примере ЧОУ ВО "Московский университет
имени С.Ю. Витте").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import pandas as pd


@dataclass
class DatasetInfo:
    """Информация о загруженном датасете."""
    path: str
    num_rows: int
    num_columns: int
    columns: List[str]
    target_column: str


def load_csv_local(path: str, target_column: str) -> Tuple[pd.DataFrame, DatasetInfo]:
    """Загрузка CSV-файла из локальной файловой системы."""
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(
            f"Целевой столбец '{target_column}' не найден в файле {path}. "
            f"Доступные столбцы: {list(df.columns)}"
        )

    info = DatasetInfo(
        path=path,
        num_rows=df.shape[0],
        num_columns=df.shape[1],
        columns=list(df.columns),
        target_column=target_column,
    )
    return df, info


def load_csv_from_url(url: str, target_column: str) -> Tuple[pd.DataFrame, DatasetInfo]:
    """Загрузка CSV-файла по URL (например, с Kaggle, GitHub, Google Drive и т.д.)."""
    df = pd.read_csv(url)
    if target_column not in df.columns:
        raise ValueError(
            f"Целевой столбец '{target_column}' не найден в данных по URL {url}. "
            f"Доступные столбцы: {list(df.columns)}"
        )

    info = DatasetInfo(
        path=url,
        num_rows=df.shape[0],
        num_columns=df.shape[1],
        columns=list(df.columns),
        target_column=target_column,
    )
    return df, info
