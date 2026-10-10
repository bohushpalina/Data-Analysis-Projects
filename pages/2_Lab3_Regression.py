import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Лабораторная 3", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRS = [ROOT / "saved_models", ROOT / "models"]

MODELS = {
    "Парная регрессия: опыт → зарплата": {
        "file": "model_1_simple_salary.pkl",
        "slug": "karthickveerakumar/salary-data-simple-linear-regression",
        "target": "Salary",
        "target_label": "Заработная плата ($)",
        "labels": {"YearsExperience": "Опыт работы (лет)"},
    },
    "Множественная А: TV → Sales": {
        "file": "model_2_multi_A.pkl",
        "slug": "ashydv/advertising-dataset",
        "target": "Sales",
        "target_label": "Объём продаж",
        "labels": {},
    },
    "Множественная Б: TV + Radio → Sales": {
        "file": "model_3_multi_B.pkl",
        "slug": "ashydv/advertising-dataset",
        "target": "Sales",
        "target_label": "Объём продаж",
        "labels": {},
    },
    "Множественная В: TV + Radio + Newspaper → Sales": {
        "file": "model_4_multi_C.pkl",
        "slug": "ashydv/advertising-dataset",
        "target": "Sales",
        "target_label": "Объём продаж",
        "labels": {},
    },
}

# запасные диапазоны, если датасет не удалось скачать
FALLBACK_RANGES = {
    "YearsExperience": (1.1, 10.5),
    "TV": (0.7, 296.4),
    "Radio": (0.0, 49.6),
    "Newspaper": (0.3, 114.0),
}


@st.cache_resource(show_spinner=False)
def load_model(filename):
    for d in MODEL_DIRS:
        p = d / filename
        if p.exists():
            return joblib.load(p)
    return None


@st.cache_data(show_spinner=False)
def load_dataset(slug):
    try:
        import kagglehub

        path = kagglehub.dataset_download(slug)
        csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
        df = pd.read_csv(os.path.join(path, csv_file))
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df
    except Exception:
        return None


st.title("Лабораторная 3 — Симулятор предсказаний (линейная регрессия)")
st.write(
    "Выберите обученную модель, задайте значения признаков ползунками слева — "
    "прогноз пересчитывается мгновенно по формуле регрессии."
)

model_name = st.selectbox("Модель", list(MODELS))
cfg = MODELS[model_name]

model = load_model(cfg["file"])
if model is None:
    st.error(
        f"Файл модели `{cfg['file']}` не найден в папке `saved_models/`. "
        "Сначала выполните ноутбук `lab3_linear_regression.ipynb` (раздел «Экспорт и сохранение моделей»)."
    )
    st.stop()

features = list(getattr(model, "feature_names_in_", []))
if not features:
    st.error("У модели нет информации об именах признаков (`feature_names_in_`).")
    st.stop()

data = load_dataset(cfg["slug"])

# ---------------------------------------------------------------- sidebar: sliders
st.sidebar.header("Значения признаков")
input_mode = st.sidebar.radio("Способ ввода", ["Ползунки", "Числовые поля"], horizontal=True)
values = {}
for f in features:
    if data is not None and f in data.columns:
        lo, hi, mean = float(data[f].min()), float(data[f].max()), float(data[f].mean())
    else:
        lo, hi = FALLBACK_RANGES.get(f, (0.0, 100.0))
        mean = (lo + hi) / 2
    label = cfg["labels"].get(f, f)
    default = round(mean, 1)
    if input_mode == "Ползунки":
        values[f] = st.sidebar.slider(label, lo, hi, default, step=0.1, key=f"slider_{f}")
    else:
        values[f] = st.sidebar.number_input(label, min_value=0.0, value=default, step=0.1,
                                            key=f"number_{f}")

# ---------------------------------------------------------------- prediction
X_new = pd.DataFrame([values], columns=features)
prediction = float(model.predict(X_new)[0])

intercept = float(model.intercept_)
coefs = dict(zip(features, [float(c) for c in model.coef_]))

col_a, col_b = st.columns([1, 2])

with col_a:
    st.metric(f"Прогноз: {cfg['target_label']}", f"{prediction:,.2f}")

    st.markdown("**Формула регрессии**")
    formula = f"ŷ = {intercept:.4f}" + "".join(f" + {w:.4f}·{f}" for f, w in coefs.items())
    st.code(formula, language=None)

    st.markdown("**Подстановка значений**")
    calc = f"ŷ = {intercept:.4f}" + "".join(
        f" + {w:.4f}·{values[f]:.2f}" for f, w in coefs.items()
    ) + f" = {prediction:.2f}"
    st.code(calc, language=None)

    with st.expander("Коэффициенты модели"):
        st.dataframe(
            pd.DataFrame({"Параметр": ["Intercept (w0)"] + features,
                          "Значение": [intercept] + list(coefs.values())}),
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------- live charts
with col_b:
    st.markdown("**График предсказаний (обновляется при движении ползунков)**")

    x_feature = features[0]
    if len(features) > 1:
        x_feature = st.selectbox("Признак по оси X (остальные фиксированы по ползункам)", features)

    if data is not None and x_feature in data.columns:
        x_min, x_max = float(data[x_feature].min()), float(data[x_feature].max())
    else:
        x_min, x_max = FALLBACK_RANGES.get(x_feature, (0.0, 100.0))

    grid = np.linspace(x_min, x_max, 100)
    grid_df = pd.DataFrame({f: (grid if f == x_feature else values[f]) for f in features},
                           columns=features)
    line = model.predict(grid_df)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(features) == 1 and data is not None and cfg["target"] in data.columns:
        ax.scatter(data[x_feature], data[cfg["target"]], alpha=0.5, color="tab:blue",
                   label="Данные датасета")
    ax.plot(grid, line, color="black", linewidth=2, label="Линия регрессии")
    ax.scatter([values[x_feature]], [prediction], color="red", s=120, zorder=5,
               label=f"Ваш прогноз: {prediction:.2f}")
    ax.set_xlabel(cfg["labels"].get(x_feature, x_feature))
    ax.set_ylabel(cfg["target_label"])
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("**Вклад каждого слагаемого в прогноз**")
parts = {"Intercept": intercept}
parts.update({f: coefs[f] * values[f] for f in features})
fig2, ax2 = plt.subplots(figsize=(8, 3))
colors = ["gray"] + ["tab:green" if v >= 0 else "tab:red" for v in list(parts.values())[1:]]
ax2.barh(list(parts.keys()), list(parts.values()), color=colors)
ax2.set_xlabel("Вклад в прогноз")
st.pyplot(fig2)
plt.close(fig2)
