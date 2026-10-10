import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Лабораторная 2", layout="wide")
sns.set_theme(style="whitegrid")

DATASETS = {
    "Math Students Performance": "adilshamim8/math-students",
    "Billionaires Statistics": "endofnight17j03/billionaires-statistics-dataset",
}

NUM_STRATEGIES = ["Заполнить медианой", "Заполнить средним", "Удалить строки", "Не обрабатывать"]
CAT_STRATEGIES = ["Заполнить 'Unknown'", "Заполнить модой", "Удалить строки", "Не обрабатывать"]


@st.cache_data(show_spinner="Загрузка датасета...")
def load_data(slug):
    import kagglehub

    path = kagglehub.dataset_download(slug)
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    return pd.read_csv(os.path.join(path, csv_file))


@st.cache_data(show_spinner=False)
def clean_data(df, drop_dups, miss_threshold, num_strategy, cat_strategy):
    """Очистка данных: дубликаты, столбцы с большим числом пропусков, заполнение пропусков."""
    df = df.copy()
    report = {}

    if drop_dups:
        before = len(df)
        df = df.drop_duplicates()
        report["Удалено дубликатов"] = before - len(df)

    miss_pct = df.isna().mean() * 100
    to_drop = miss_pct[miss_pct > miss_threshold].index.tolist()
    df = df.drop(columns=to_drop)
    report["Удалено столбцов"] = ", ".join(to_drop) if to_drop else "—"

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    rows_before = len(df)
    if num_strategy == "Заполнить медианой":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
    elif num_strategy == "Заполнить средним":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].mean())
    elif num_strategy == "Удалить строки":
        df = df.dropna(subset=num_cols)

    if cat_strategy == "Заполнить 'Unknown'":
        for c in cat_cols:
            df[c] = df[c].fillna("Unknown")
    elif cat_strategy == "Заполнить модой":
        for c in cat_cols:
            mode = df[c].mode()
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])
    elif cat_strategy == "Удалить строки":
        df = df.dropna(subset=cat_cols)

    report["Удалено строк из-за пропусков"] = rows_before - len(df)
    return df.reset_index(drop=True), report


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


st.title("Лабораторная 2 — Разведочный анализ данных")

st.sidebar.header("1. Датасет")
ds_name = st.sidebar.selectbox("Выберите датасет", list(DATASETS))

try:
    raw = load_data(DATASETS[ds_name])
except Exception as e:  # нет интернета / нет доступа к kaggle
    st.error(f"Не удалось загрузить датасет: {e}")
    st.stop()

st.sidebar.header("2. Очистка данных")
drop_dups = st.sidebar.checkbox("Удалить дубликаты", value=True)
miss_threshold = st.sidebar.slider("Удалять столбцы, где пропусков больше (%)", 0, 100, 70)
num_strategy = st.sidebar.selectbox("Пропуски в числовых признаках", NUM_STRATEGIES)
cat_strategy = st.sidebar.selectbox("Пропуски в категориальных признаках", CAT_STRATEGIES)

df, report = clean_data(raw, drop_dups, miss_threshold, num_strategy, cat_strategy)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]
low_card_cols = [c for c in cat_cols if df[c].nunique() <= 50]

st.sidebar.header("3. Фильтры")
mask = pd.Series(True, index=df.index)

filter_num = st.sidebar.multiselect("Числовые фильтры (столбцы)", num_cols)
for c in filter_num:
    lo, hi = float(df[c].min()), float(df[c].max())
    if lo < hi:
        rng = st.sidebar.slider(c, lo, hi, (lo, hi))
        mask &= df[c].between(rng[0], rng[1])

filter_cat = st.sidebar.multiselect("Категориальные фильтры (столбцы)", low_card_cols)
for c in filter_cat:
    options = sorted(df[c].dropna().astype(str).unique())
    chosen = st.sidebar.multiselect(c, options, default=options)
    mask &= df[c].astype(str).isin(chosen)

fdf = df[mask]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Строк (исходно)", f"{len(raw):,}")
c2.metric("Строк после очистки", f"{len(df):,}")
c3.metric("Строк после фильтров", f"{len(fdf):,}")
c4.metric("Столбцов", len(df.columns))

with st.expander("Отчёт об очистке"):
    for k, v in report.items():
        st.write(f"**{k}:** {v}")

if fdf.empty:
    st.warning("После применения фильтров не осталось данных. Ослабьте фильтры.")
    st.stop()

tab_data, tab_stats, tab_plots = st.tabs(["Таблица", "Статистика и пропуски", "Графики"])

with tab_data:
    cols = st.multiselect("Показать столбцы", list(fdf.columns), default=list(fdf.columns))
    view = st.radio("Какие строки показать", ["Первые", "Последние", "Случайные"], horizontal=True)
    n = st.slider("Количество строк", 5, 100, 10)
    if view == "Первые":
        shown = fdf[cols].head(n)
    elif view == "Последние":
        shown = fdf[cols].tail(n)
    else:
        shown = fdf[cols].sample(min(n, len(fdf)), random_state=42)
    st.dataframe(shown, use_container_width=True)
    st.download_button(
        " Скачать отфильтрованные данные (CSV)",
        fdf.to_csv(index=False).encode("utf-8"),
        file_name="filtered_data.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------- stats tab
with tab_stats:
    st.subheader("Числовые признаки")
    if num_cols:
        st.dataframe(
            fdf[num_cols].describe().T[["mean", "50%", "std", "min", "max"]].rename(
                columns={"mean": "среднее", "50%": "медиана", "std": "станд. отклонение",
                         "min": "минимум", "max": "максимум"}
            ),
            use_container_width=True,
        )
    st.subheader("Категориальные признаки")
    if cat_cols:
        st.dataframe(
            fdf[cat_cols].describe().T[["unique", "top", "freq"]].rename(
                columns={"unique": "уникальных значений", "top": "мода", "freq": "частота моды"}
            ),
            use_container_width=True,
        )
    st.subheader("Пропуски по столбцам (в отфильтрованных данных)")
    miss = fdf.isna().sum().rename("пропусков").to_frame()
    miss["%"] = (miss["пропусков"] / len(fdf) * 100).round(2)
    st.dataframe(miss[miss["пропусков"] > 0] if miss["пропусков"].sum() else miss.head(0),
                 use_container_width=True)
    if miss["пропусков"].sum() == 0:
        st.success("Пропусков нет.")
    st.write(f"**Дубликатов строк:** {int(fdf.duplicated().sum())}")

# ---------------------------------------------------------------- plots tab
with tab_plots:
    chart = st.selectbox(
        "Тип графика",
        ["Гистограмма", "График плотности (KDE)", "Boxplot (ящик с усами)", "Violin plot",
         "Столбчатая диаграмма (категории)", "Круговая диаграмма", "Диаграмма рассеяния (scatter)",
         "Тепловая карта корреляций"],
    )

    hue_options = ["— нет —"] + low_card_cols

    if chart in ("Гистограмма", "График плотности (KDE)", "Boxplot (ящик с усами)", "Violin plot"):
        if not num_cols:
            st.warning("Нет числовых столбцов.")
            st.stop()
        col = st.selectbox("Числовой признак", num_cols)
        group = st.selectbox("Разбить по категории (необязательно)", hue_options)
        hue = None if group == "— нет —" else group
        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            if chart == "Гистограмма":
                bins = st.slider("Количество корзин (bins)", 5, 100, 30)
                show_kde = st.checkbox("Показать KDE", value=True)
                log_x = st.checkbox("Логарифмическая шкала X", value=False)
                sns.histplot(data=fdf, x=col, hue=hue, bins=bins, kde=show_kde, log_scale=log_x, ax=ax)
            elif chart == "График плотности (KDE)":
                sns.kdeplot(data=fdf, x=col, hue=hue, fill=True, common_norm=False, ax=ax)
            elif chart == "Boxplot (ящик с усами)":
                sns.boxplot(data=fdf, x=col, y=hue, orient="h", ax=ax)
            else:
                sns.violinplot(data=fdf, x=col, y=hue, orient="h", ax=ax)
            ax.set_title(f"{chart}: {col}")
            show_fig(fig)
        except Exception as e:
            plt.close(fig)
            st.error(f"Не удалось построить график: {e}")

    elif chart in ("Столбчатая диаграмма (категории)", "Круговая диаграмма"):
        if not cat_cols:
            st.warning("Нет категориальных столбцов.")
            st.stop()
        col = st.selectbox("Категориальный признак", cat_cols)
        top_n = st.slider("Показать top-N категорий", 2, 15, 5)
        counts = fdf[col].astype(str).value_counts()
        top = counts.head(top_n)
        others = counts.iloc[top_n:].sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        if chart == "Столбчатая диаграмма (категории)":
            sns.barplot(x=top.values, y=top.index, ax=ax)
            ax.set_xlabel("Количество")
            ax.set_ylabel(col)
        else:
            pie_data = top.copy()
            if others > 0:
                pie_data["Остальные"] = others
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
        ax.set_title(f"{chart}: {col}")
        show_fig(fig)

    elif chart == "Диаграмма рассеяния (scatter)":
        if len(num_cols) < 2:
            st.warning("Нужно минимум два числовых столбца.")
            st.stop()
        cc1, cc2 = st.columns(2)
        x = cc1.selectbox("Ось X", num_cols, index=0)
        y = cc2.selectbox("Ось Y", num_cols, index=min(1, len(num_cols) - 1))
        group = st.selectbox("Цвет по категории (необязательно)", hue_options)
        hue = None if group == "— нет —" else group
        lc1, lc2 = st.columns(2)
        log_x = lc1.checkbox("Логарифмическая шкала X")
        log_y = lc2.checkbox("Логарифмическая шкала Y")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=fdf, x=x, y=y, hue=hue, ax=ax)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(f"{y} vs {x}")
        show_fig(fig)
        st.write(f"Коэффициент корреляции Пирсона: **{fdf[x].corr(fdf[y]):.3f}**")

    else:  # Тепловая карта
        if len(num_cols) < 2:
            st.warning("Нужно минимум два числовых столбца.")
            st.stop()
        chosen = st.multiselect("Признаки для матрицы корреляции", num_cols, default=num_cols[:8])
        if len(chosen) < 2:
            st.info("Выберите хотя бы два признака.")
        else:
            fig, ax = plt.subplots(figsize=(max(6, len(chosen)), max(5, len(chosen) * 0.7)))
            sns.heatmap(fdf[chosen].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                        vmin=-1, vmax=1, ax=ax)
            ax.set_title("Тепловая карта корреляций")
            show_fig(fig)
