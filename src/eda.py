# -*- coding: utf-8 -*-
"""
Разведочный анализ данных (EDA).

Автор: Ревнивцев Артем Александрович
Тема ВКР: Интеллектуальная система прогнозирования потребностей в обновлении вычислительной техники.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_ticket_counts_by_department(df: pd.DataFrame) -> plt.Figure:
    """Количество обращений по подразделениям."""
    counts = df["user_department"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Подразделение")
    ax.set_ylabel("Количество обращений")
    ax.set_title("Распределение обращений по подразделениям")
    fig.tight_layout()
    return fig


def plot_ticket_counts_by_device_type(df: pd.DataFrame) -> plt.Figure:
    """Количество обращений по типам устройств."""
    counts = df["device_type"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Тип устройства")
    ax.set_ylabel("Количество обращений")
    ax.set_title("Распределение обращений по типам устройств")
    fig.tight_layout()
    return fig


def plot_device_age_hist(df: pd.DataFrame) -> plt.Figure:
    """Гистограмма возраста устройств."""
    fig, ax = plt.subplots()
    df["device_age_years"].hist(ax=ax, bins=10)
    ax.set_xlabel("Возраст устройства, лет")
    ax.set_ylabel("Количество устройств")
    ax.set_title("Распределение возраста устройств")
    fig.tight_layout()
    return fig


def plot_tickets_last_6_months_hist(df: pd.DataFrame) -> plt.Figure:
    """Гистограмма количества обращений за последние 6 месяцев."""
    fig, ax = plt.subplots()
    df["tickets_last_6_months"].hist(ax=ax, bins=10)
    ax.set_xlabel("Обращений за 6 месяцев")
    ax.set_ylabel("Количество устройств")
    ax.set_title("Распределение обращений за 6 месяцев")
    fig.tight_layout()
    return fig
