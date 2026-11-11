import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Лабораторная 5", layout="wide")
sns.set_theme(style="whitegrid")

try:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
except ImportError as e:
    st.error(f"Не установлена библиотека: {e}. Выполните: pip install xgboost lightgbm")
    st.stop()

SEED = 42

DATASETS = {
    "Датасет №1: Pima Indians Diabetes": "salihacur/diabetes",
    "Датасет №2: Churn Modelling": "shrutimechlearn/churn-modelling",
}

CLASSIC = ["kNN", "Decision Tree", "SVM", "MLP"]
ENSEMBLES = ["Bagging", "LightGBM", "XGBoost", "Stacking"]
ALL_MODELS = CLASSIC + ENSEMBLES
DEFAULT_ON = {"SVM", "MLP", "LightGBM", "XGBoost"}

# фиксированные цвета, чтобы модель имела один и тот же цвет на всех графиках
PALETTE = dict(zip(ALL_MODELS, sns.color_palette("tab10", len(ALL_MODELS))))

# лучшие гиперпараметры: классика из ЛР №4 (GridSearchCV), ансамбли из ЛР №5 (cross_valid)
PARAMS = {
    "df1": {
        "kNN": dict(n_neighbors=11),
        "Decision Tree": dict(max_depth=3),
        "SVM": dict(C=0.1, kernel="linear"),
        "MLP": dict(hidden_layer_sizes=(50,), activation="relu"),
        "Bagging": dict(n_estimators=75),
        "LightGBM": dict(n_estimators=50, learning_rate=0.03, max_depth=4),
        "XGBoost": dict(n_estimators=300, learning_rate=0.05, max_depth=2),
        "Stacking": dict(C=1),
    },
    "df2": {
        "kNN": dict(n_neighbors=9),
        "Decision Tree": dict(max_depth=5),
        "SVM": dict(C=1, kernel="rbf"),
        "MLP": dict(hidden_layer_sizes=(50,), activation="relu"),
        "Bagging": dict(n_estimators=75),
        "LightGBM": dict(n_estimators=50, learning_rate=0.05, max_depth=5),
        "XGBoost": dict(n_estimators=50, learning_rate=0.05, max_depth=5),
        "Stacking": dict(C=0.1),
    },
}


def read_csv_from_kaggle(slug):
    import kagglehub

    path = kagglehub.dataset_download(slug)
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    return pd.read_csv(os.path.join(path, csv_file))


@st.cache_resource(show_spinner="Загрузка и предобработка датасета...")
def prepare_dataset(ds_name):
    """Та же предобработка, что в ноутбуке. Возвращает выборки и функцию transform
    для обработки «сырых» строк, загруженных пользователем."""
    df = read_csv_from_kaggle(DATASETS[ds_name])
    scaler = StandardScaler()

    if ds_name.startswith("Датасет №1"):
        ds_key, target = "df1", "Outcome"
        class_names = ["Нет диабета", "Диабет"]
        zero_missing = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}
        cols = [c for c in df.columns if c.lower() in zero_missing]

        clean = df.copy()
        clean[cols] = clean[cols].replace(0, np.nan)
        X, y = clean.drop(columns=target), clean[target]
        feature_names = list(X.columns)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        test_index = X_te.index

        imputer = SimpleImputer(strategy="median")
        X_tr = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_te = scaler.transform(imputer.transform(X_te))

        def transform(raw_df):
            d = raw_df.copy()
            d[cols] = d[cols].replace(0, np.nan)
            return scaler.transform(imputer.transform(d[feature_names]))
    else:
        ds_key, target = "df2", "Exited"
        class_names = ["Остался", "Ушёл"]
        drop_cols = ["RowNumber", "CustomerId", "Surname"]
        enc = pd.get_dummies(df.drop(columns=drop_cols), columns=["Geography", "Gender"],
                             drop_first=True, dtype=int)
        X, y = enc.drop(columns=target), enc[target]
        feature_names = list(X.columns)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        test_index = X_te.index

        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        def transform(raw_df):
            d = raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns])
            d = pd.get_dummies(d, columns=["Geography", "Gender"], dtype=int)
            return scaler.transform(d.reindex(columns=feature_names, fill_value=0))

    return {
        "key": ds_key,
        "target": target,
        "class_names": class_names,
        "X_tr": X_tr, "y_tr": y_tr.to_numpy(),
        "X_te": X_te, "y_te": y_te.to_numpy(),
        "raw_test": df.loc[test_index],
        "transform": transform,
    }


def build_model(name, ds_key):
    p = PARAMS[ds_key]
    if name == "kNN":
        return KNeighborsClassifier(**p[name])
    if name == "Decision Tree":
        return DecisionTreeClassifier(**p[name], random_state=SEED)
    if name == "SVM":
        return SVC(**p[name])
    if name == "MLP":
        return MLPClassifier(**p[name], max_iter=1000, random_state=SEED)
    if name == "Bagging":
        return BaggingClassifier(**p[name], random_state=SEED)
    if name == "LightGBM":
        return LGBMClassifier(**p[name], random_state=SEED, verbose=-1)
    if name == "XGBoost":
        return XGBClassifier(**p[name], random_state=SEED, verbosity=0)
    base = [
        ("svm", make_pipeline(StandardScaler(), SVC(**p["SVM"], probability=True, random_state=SEED))),
        ("mlp", make_pipeline(StandardScaler(), MLPClassifier(**p["MLP"], max_iter=1000, random_state=SEED))),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(C=p["Stacking"]["C"], max_iter=1000),
        cv=5,
    )


@st.cache_resource(show_spinner=False)
def fit_model(ds_name, model_name):
    data = prepare_dataset(ds_name)
    model = build_model(model_name, data["key"])
    model.fit(data["X_tr"], data["y_tr"])
    return model


def get_scores(model, X):
    """Оценка принадлежности к классу 1 для ROC: вероятность или решающая функция (SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y):
    pred = model.predict(X)
    score = get_scores(model, X)
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, score),
        "cm": confusion_matrix(y, pred),
        "fpr": fpr,
        "tpr": tpr,
    }


def params_to_text(d):
    return ", ".join(f"{k}={v}" for k, v in d.items())



st.title("Лабораторная 5 — Классика vs Ансамбли")
st.write(
    "Интерактивный бенчмарк: отметьте галочками классические и ансамблевые модели, "
    "и они будут сравнены на одной тестовой выборке — по ROC-кривым и столбчатым диаграммам метрик."
)

st.sidebar.header("1. Датасет")
ds_name = st.sidebar.selectbox("Выберите датасет", list(DATASETS))

try:
    data = prepare_dataset(ds_name)
except Exception as e:
    st.error(f"Не удалось загрузить датасет: {e}")
    st.stop()

ds_key, class_names = data["key"], data["class_names"]

st.sidebar.header("2. Тестовое подмножество")
source = st.sidebar.radio("Источник", ["Тестовая выборка из ноутбука (20%)", "Загрузить свой CSV"])

if source.startswith("Тестовая"):
    share = st.sidebar.slider("Доля тестовой выборки, %", 20, 100, 100, step=10)
    X_test, y_test = data["X_te"], data["y_te"]
    if share < 100:
        _, idx = train_test_split(np.arange(len(y_test)), test_size=share / 100,
                                  stratify=y_test, random_state=SEED)
        X_test, y_test = X_test[idx], y_test[idx]
    st.sidebar.download_button(
        "Скачать тестовую выборку (CSV)",
        data=data["raw_test"].to_csv(index=False).encode("utf-8"),
        file_name=f"test_{ds_key}.csv",
        mime="text/csv",
    )
else:
    uploaded = st.sidebar.file_uploader("CSV с теми же столбцами, что в исходном датасете", type="csv")
    st.sidebar.caption(
        "Файл должен содержать целевой столбец "
        f"«{data['target']}». Строки из обучающей выборки завысят метрики."
    )
    if uploaded is None:
        st.info("Загрузите CSV в боковой панели или вернитесь к тестовой выборке из ноутбука. "
                "Пример подходящего файла можно скачать при выборе первого варианта.")
        st.stop()
    try:
        raw_up = pd.read_csv(uploaded)
        if data["target"] not in raw_up.columns:
            raise ValueError(f"нет целевого столбца «{data['target']}»")
        y_test = raw_up[data["target"]].to_numpy()
        X_test = data["transform"](raw_up.drop(columns=data["target"]))
    except Exception as e:
        st.error(f"Не удалось обработать файл: {e}")
        st.stop()

if len(np.unique(y_test)) < 2:
    st.error("В тестовом подмножестве должны быть оба класса — иначе ROC-кривая не определена.")
    st.stop()

st.sidebar.header("3. Модели для сравнения")
selected = []
st.sidebar.subheader("Классические (ЛР №4)")
for name in CLASSIC:
    if st.sidebar.checkbox(name, value=name in DEFAULT_ON, key=f"cb_{name}"):
        selected.append(name)
st.sidebar.subheader("Ансамбли (ЛР №5)")
for name in ENSEMBLES:
    if st.sidebar.checkbox(name, value=name in DEFAULT_ON, key=f"cb_{name}"):
        selected.append(name)

st.sidebar.header("4. Тип графика")
chart_type = st.sidebar.radio("Что показывать", ["ROC-кривые", "Столбчатые диаграммы метрик", "Оба графика"])

if not selected:
    st.info("Отметьте хотя бы одну модель в боковой панели.")
    st.stop()

if "Stacking" in selected and ds_key == "df2":
    st.caption("Стэкинг на втором датасете обучается заметно дольше остальных (при первом запуске), "
               "затем модель берётся из кэша.")

results = {}
for name in selected:
    with st.spinner(f"Обучение модели {name}..."):
        model = fit_model(ds_name, name)
    results[name] = evaluate(model, X_test, y_test)

st.subheader(f"{len(selected)} моделей на «{ds_name}»")
st.caption(f"Объектов в тестовом подмножестве: {len(y_test)}, доля класса «1»: {np.mean(y_test):.2f}. "
           "Precision, Recall и F1 считаются для класса «1».")

# --- таблица метрик
metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
table = pd.DataFrame(
    [
        {"Модель": n, "Тип": "Классика" if n in CLASSIC else "Ансамбль", **{m: r[m] for m in metric_cols}}
        for n, r in results.items()
    ]
).sort_values("F1", ascending=False)
st.dataframe(table.round(4), use_container_width=True, hide_index=True)


# --- графики
def draw_roc():
    fig, ax = plt.subplots(figsize=(8, 6))
    for n, r in results.items():
        ax.plot(r["fpr"], r["tpr"], color=PALETTE[n], linewidth=2,
                linestyle="--" if n in CLASSIC else "-",
                label=f"{n} (AUC = {r['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Случайный классификатор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC-кривые: {ds_name}\nпунктир — классика, сплошная — ансамбли")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def draw_bars():
    long = table.melt(id_vars=["Модель", "Тип"], value_vars=metric_cols,
                      var_name="Метрика", value_name="Значение")
    order = list(table["Модель"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=long, x="Метрика", y="Значение", hue="Модель", hue_order=order,
                palette=PALETTE, ax=ax)
    for container, n in zip(ax.containers, order):
        if n in CLASSIC:
            for bar in container:
                bar.set_hatch("//")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Метрики качества: {ds_name}\nштриховка — классика, без штриховки — ансамбли")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


if chart_type in ("ROC-кривые", "Оба графика"):
    fig = draw_roc()
    st.pyplot(fig)
    plt.close(fig)

if chart_type in ("Столбчатые диаграммы метрик", "Оба графика"):
    fig = draw_bars()
    st.pyplot(fig)
    plt.close(fig)

# --- матрицы неточностей
with st.expander("Матрицы неточностей выбранных моделей"):
    n = len(results)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Гиперпараметры моделей (ЛР №4 — GridSearchCV, ЛР №5 — кросс-валидация)"):
    st.dataframe(
        pd.DataFrame(
            [
                {"Модель": n, "Тип": "Классика" if n in CLASSIC else "Ансамбль",
                 "Параметры": params_to_text(PARAMS[ds_key][n]) +
                              ("  (мета-модель: LogisticRegression; базовые: SVM, MLP)" if n == "Stacking" else "")}
                for n in ALL_MODELS
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
