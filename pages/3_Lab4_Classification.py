import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(page_title="Лабораторная 4", layout="wide")
sns.set_theme(style="whitegrid")

DATASETS = {
    "Датасет №1: Pima Indians Diabetes": "salihacur/diabetes",
    "Датасет №2: Churn Modelling": "shrutimechlearn/churn-modelling",
}

BEST_FROM_NOTEBOOK = pd.DataFrame(
    {
        "Алгоритм": ["kNN", "Decision Tree", "SVM", "MLP"],
        "Датасет №1 (диабет)": ["11 соседей", "глубина 3", "C=0.1, ядро linear", "1 слой (50), relu"],
        "Датасет №2 (отток)": ["9 соседей", "глубина 5", "C=1, ядро rbf", "1 слой (50), relu"],
    }
)


def read_csv_from_kaggle(slug):
    import kagglehub

    path = kagglehub.dataset_download(slug)
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    return pd.read_csv(os.path.join(path, csv_file))


@st.cache_data(show_spinner="Загрузка и предобработка датасета...")
def prepare_dataset(ds_name):
    """Предобработка точно как в ноутбуке: пропуски, кодирование, stratify-split, StandardScaler."""
    df = read_csv_from_kaggle(DATASETS[ds_name])

    if ds_name.startswith("Датасет №1"):
        class_names = ["Нет диабета", "Диабет"]
        # целевой столбец в этом датасете называется Outcome
        target = "Outcome"
        # нули в этих признаках означают пропуски (Pregnancies не трогаем: там 0 возможен)
        zero_missing = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}
        cols = [c for c in df.columns if c.lower() in zero_missing]
        df[cols] = df[cols].replace(0, np.nan)
        X, y = df.drop(columns=target), df[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)
    else:
        class_names = ["Остался", "Ушёл"]
        df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
        df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True, dtype=int)
        X, y = df.drop(columns="Exited"), df["Exited"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy(), class_names


def build_model(name, p):
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"])
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=p["max_depth"], criterion=p["criterion"], random_state=42)
    if name == "SVM":
        return SVC(C=p["C"], kernel=p["kernel"])
    return MLPClassifier(hidden_layer_sizes=p["hidden_layer_sizes"], activation=p["activation"],
                         max_iter=1000, random_state=42)


@st.cache_data(show_spinner="Обучение модели...")
def train_and_evaluate(ds_name, model_name, params_items):
    X_tr, X_te, y_tr, y_te, _ = prepare_dataset(ds_name)
    model = build_model(model_name, dict(params_items))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "cm": confusion_matrix(y_te, pred),
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred, zero_division=0),
        "Train accuracy": accuracy_score(y_tr, model.predict(X_tr)),
    }


st.title("Лабораторная 4 — Классификация")
st.write(
    "Выберите датасет и классификатор, меняйте гиперпараметры в боковой панели — "
    "модель переобучается, а матрица неточностей и метрики обновляются сразу."
)

st.sidebar.header("1. Датасет")
ds_name = st.sidebar.selectbox("Выберите датасет", list(DATASETS))

try:
    X_tr, X_te, y_tr, y_te, class_names = prepare_dataset(ds_name)
except Exception as e:
    st.error(f"Не удалось загрузить датасет: {e}")
    st.stop()

st.sidebar.header("2. Классификатор")
model_name = st.sidebar.selectbox("Выберите классификатор", ["kNN", "Decision Tree", "SVM", "MLP"])

st.sidebar.header("3. Гиперпараметры")
if model_name == "kNN":
    params = {
        "n_neighbors": st.sidebar.slider("n_neighbors (число соседей)", 1, 31, 5),
        "weights": st.sidebar.selectbox("weights", ["uniform", "distance"]),
    }
elif model_name == "Decision Tree":
    unlimited = st.sidebar.checkbox("Без ограничения глубины (max_depth=None)", value=False)
    depth = st.sidebar.slider("max_depth (глубина дерева)", 1, 20, 5, disabled=unlimited)
    params = {
        "max_depth": None if unlimited else depth,
        "criterion": st.sidebar.selectbox("criterion", ["gini", "entropy"]),
    }
elif model_name == "SVM":
    params = {
        "C": st.sidebar.select_slider("C (регуляризация)", options=[0.01, 0.1, 1, 10, 100], value=1),
        "kernel": st.sidebar.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"]),
    }
else:
    n_layers = st.sidebar.slider("Число скрытых слоёв", 1, 3, 1)
    n_neurons = st.sidebar.slider("Нейронов в каждом слое", 5, 200, 50, step=5)
    params = {
        "hidden_layer_sizes": tuple([n_neurons] * n_layers),
        "activation": st.sidebar.selectbox("activation", ["relu", "tanh", "logistic"]),
    }

result = train_and_evaluate(ds_name, model_name, tuple(sorted(params.items(), key=lambda kv: kv[0])))

st.subheader(f"{model_name} на «{ds_name}»")
st.caption("Параметры: " + ", ".join(f"{k}={v}" for k, v in params.items()))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{result['Accuracy']:.3f}")
m2.metric("Precision", f"{result['Precision']:.3f}")
m3.metric("Recall", f"{result['Recall']:.3f}")
m4.metric("F1-score", f"{result['F1']:.3f}")

left, right = st.columns(2)

with left:
    st.markdown("**Матрица неточностей (тестовая выборка)**")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(result["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Таблица метрик (метрики считаются для класса «1»)**")
    metrics_table = pd.DataFrame(
        {
            "Метрика": ["Accuracy", "Precision", "Recall", "F1-score", "Accuracy на train"],
            "Значение": [result["Accuracy"], result["Precision"], result["Recall"],
                         result["F1"], result["Train accuracy"]],
        }
    )
    st.dataframe(metrics_table.round(4), use_container_width=True, hide_index=True)
    st.caption(
        "Если Accuracy на train заметно выше, чем на test — модель переобучена. "
        f"Доля класса «1» в тесте: {y_te.mean():.2f}, поэтому при несбалансированных классах "
        "ориентируйтесь на Recall и F1."
    )

st.markdown("---")
st.subheader("Сравнение конфигураций")

if "history" not in st.session_state:
    st.session_state["history"] = []

b1, b2 = st.columns([1, 1])
if b1.button("Добавить текущую модель в таблицу сравнения"):
    st.session_state["history"].append(
        {
            "Датасет": ds_name,
            "Модель": model_name,
            "Параметры": ", ".join(f"{k}={v}" for k, v in params.items()),
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1": result["F1"],
        }
    )
if b2.button("Очистить таблицу"):
    st.session_state["history"] = []

hist = pd.DataFrame(st.session_state["history"])
if not hist.empty:
    hist = hist[hist["Датасет"] == ds_name].drop(columns="Датасет")
if hist.empty:
    st.info("Добавьте несколько конфигураций кнопкой выше, чтобы сравнить их между собой.")
else:
    st.dataframe(hist.round(4), use_container_width=True, hide_index=True)

with st.expander("Лучшие гиперпараметры, найденные в ноутбуке (GridSearchCV, 5-fold)"):
    st.dataframe(BEST_FROM_NOTEBOOK, use_container_width=True, hide_index=True)
