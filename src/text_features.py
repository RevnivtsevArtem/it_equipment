# -*- coding: utf-8 -*-
"""
Извлечение текстовых признаков из описаний обращений.

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_text_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Создаёт TF-IDF векторизатор для поля problem_description."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    return vectorizer


def fit_text_vectorizer(df: pd.DataFrame, text_column: str) -> Tuple[TfidfVectorizer, any]:
    """Обучает TF-IDF на текстовом столбце и возвращает матрицу признаков."""
    if text_column not in df.columns:
        raise ValueError(f"Столбец с текстом '{text_column}' не найден.")

    vectorizer = build_text_vectorizer()
    X_text = vectorizer.fit_transform(df[text_column].fillna(""))
    return vectorizer, X_text
