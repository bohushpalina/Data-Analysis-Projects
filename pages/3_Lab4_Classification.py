import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgba
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(page_title="Лабораторная 4", layout="wide")
sns.set_theme(style="whitegrid")

COLORS = ["tab:blue", "tab:red"]

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
    df = read_csv_from_kaggle(DATASETS[ds_name])

    if ds_name.startswith("Датасет №1"):
        class_names = ["Нет диабета", "Диабет"]
        target = "Outcome"
        zero_missing = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}
        cols = [c for c in df.columns if c.lower() in zero_missing]
        df[cols] = df[cols].replace(0, np.nan)
        X, y = df.drop(columns=target), df[target]
        feature_names = list(X.columns)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)
    else:
        class_names = ["Остался", "Ушёл"]
        df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
        df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True, dtype=int)
        X, y = df.drop(columns="Exited"), df["Exited"]
        feature_names = list(X.columns)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy(), class_names, feature_names


def build_model(name, p):
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"])
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=p["max_depth"], criterion=p["criterion"], random_state=42)
    if name == "SVM":
        return SVC(C=p["C"], kernel=p["kernel"])
    return MLPClassifier(hidden_layer_sizes=p["hidden_layer_sizes"], activation=p["activation"],
                         max_iter=1000, random_state=42)


@st.cache_resource(show_spinner=False)
def fit_model(ds_name, model_name, params_items):
    X_tr, _, y_tr, _, _, _ = prepare_dataset(ds_name)
    model = build_model(model_name, dict(params_items))
    model.fit(X_tr, y_tr)
    return model


@st.cache_data(show_spinner="Обучение модели...")
def train_and_evaluate(ds_name, model_name, params_items):
    X_tr, X_te, y_tr, y_te, _, _ = prepare_dataset(ds_name)
    model = fit_model(ds_name, model_name, params_items)
    pred = model.predict(X_te)
    return {
        "cm": confusion_matrix(y_te, pred),
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred, zero_division=0),
        "Train accuracy": accuracy_score(y_tr, model.predict(X_tr)),
    }


def params_to_text(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def plot_boundary(model_name, params, X_tr, y_tr, X_te, y_te, class_names, ds_name):
    pca = PCA(n_components=2, random_state=42)
    X_tr_2d = pca.fit_transform(X_tr)
    X_te_2d = pca.transform(X_te)
    model = build_model(model_name, params).fit(X_tr_2d, y_tr)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    DecisionBoundaryDisplay.from_estimator(
        model, X_tr_2d, ax=ax, cmap=ListedColormap(COLORS), alpha=0.25,
        response_method="predict", grid_resolution=150, plot_method="contourf",
    )
    for cls, label in enumerate(class_names):
        mask = y_te == cls
        ax.scatter(X_te_2d[mask, 0], X_te_2d[mask, 1], color=COLORS[cls], label=label,
                   edgecolor="k", linewidth=0.4, s=32, alpha=0.7)
    ax.set_title(f"{model_name}: граница решений, {ds_name}\nПараметры: {params_to_text(params)}")
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_decision_tree(model, feature_names, class_names, params, ds_name):
    fig, ax = plt.subplots(figsize=(28, 12))
    plot_tree(model, max_depth=3, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=18, ax=ax)
    ax.set_title(f"Decision Tree, {ds_name}: {params_to_text(params)}", fontsize=16, pad=20)
    fig.tight_layout()
    return fig


def plot_mlp(model, params, ds_name, max_nodes=50):
    coefs = model.coefs_
    sizes = [coefs[0].shape[0]] + [c.shape[1] for c in coefs]
    shown = [min(s, max_nodes) for s in sizes]
    indices = [np.linspace(0, s - 1, n, dtype=int) for s, n in zip(sizes, shown)]

    weights = [coefs[k][np.ix_(indices[k], indices[k + 1])] for k in range(len(coefs))]
    max_w = max(max(np.max(np.abs(w)) for w in weights), 1e-12)

    max_span = max(shown) / 2
    ys = [np.linspace(-max_span, max_span, n) if n > 1 else np.array([0.0]) for n in shown]

    segments, line_colors, widths = [], [], []
    for k, w in enumerate(weights):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                strength = abs(w[i, j]) / max_w
                segments.append([(k, ys[k][i]), (k + 1, ys[k + 1][j])])
                line_colors.append(to_rgba("blue" if w[i, j] >= 0 else "red", 0.15 + 0.8 * strength))
                widths.append(0.2 + 2.5 * strength)

    fig, ax = plt.subplots(figsize=(14, 18))
    ax.add_collection(LineCollection(segments, colors=line_colors, linewidths=widths, zorder=1))

    node_size = min(850, 9000 / max(shown))
    font_size = 10 if max(shown) <= 25 else 6
    last = len(sizes) - 1
    for k in range(len(sizes)):
        ax.scatter([k] * shown[k], ys[k], s=node_size, color="#8ab4f8",
                   edgecolor="black", linewidth=1.2, zorder=2)
        for pos, y in zip(indices[k], ys[k]):
            if k == 0:
                text = f"x{pos + 1}"
            elif k == last:
                text = "y"
            else:
                text = f"h{pos + 1}"
            ax.text(k, y, text, ha="center", va="center", fontsize=font_size,
                    fontweight="bold", zorder=3)

    for k in range(len(sizes)):
        if k == 0:
            label = f"Вход\n{sizes[0]} признаков"
        elif k == last:
            label = "Выход\n1 нейрон"
        else:
            label = f"Скрытый слой {k}\n{shown[k]} из {sizes[k]} нейронов\n{model.activation}"
        ax.text(k, -max_span - 2.2, label, ha="center", va="top", fontsize=12)

    ax.set_xlim(-0.5, last + 0.5)
    ax.set_ylim(-max_span - 4.0, max_span + 1.5)
    ax.set_title(
        f"MLP архитектура: {ds_name}\nПараметры: {params_to_text(params)}\n"
        "Синие — положительные веса, красные — отрицательные",
        fontsize=14,
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner="Строим график модели...")
def build_model_figure(ds_name, model_name, params_items):
    params = dict(params_items)
    X_tr, X_te, y_tr, y_te, class_names, feature_names = prepare_dataset(ds_name)
    if model_name in ("kNN", "SVM"):
        fig = plot_boundary(model_name, params, X_tr, y_tr, X_te, y_te, class_names, ds_name)
    elif model_name == "Decision Tree":
        model = fit_model(ds_name, model_name, params_items)
        fig = plot_decision_tree(model, feature_names, class_names, params, ds_name)
    else:
        model = fit_model(ds_name, model_name, params_items)
        fig = plot_mlp(model, params, ds_name)
    plt.close(fig)
    return fig


st.title("Лабораторная 4 — Классификация")
st.write(
    "Выберите датасет и классификатор, меняйте гиперпараметры в боковой панели — "
    "модель переобучается, а матрица неточностей, метрики и график модели обновляются сразу."
)

st.sidebar.header("1. Датасет")
ds_name = st.sidebar.selectbox("Выберите датасет", list(DATASETS))

try:
    X_tr, X_te, y_tr, y_te, class_names, feature_names = prepare_dataset(ds_name)
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

params_items = tuple(sorted(params.items(), key=lambda kv: kv[0]))
result = train_and_evaluate(ds_name, model_name, params_items)

st.subheader(f"{model_name} на «{ds_name}»")
st.caption("Параметры: " + params_to_text(params))

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
st.subheader("Графическое отображение модели")

if model_name in ("kNN", "SVM"):
    st.caption(
        "Признаки сжаты до двух главных компонент (PCA), модель переобучена на них. "
        "Закрашены области классов, точками показана тестовая выборка."
    )
elif model_name == "Decision Tree":
    st.caption("Показаны первые 3 уровня построенного дерева.")
else:
    st.caption("Схема обученной сети: толщина линий отражает величину весов.")

st.pyplot(build_model_figure(ds_name, model_name, params_items))

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
            "Параметры": params_to_text(params),
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
