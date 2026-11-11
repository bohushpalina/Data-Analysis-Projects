import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
import streamlit as st
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Лабораторная 6", layout="wide")
sns.set_theme(style="whitegrid")

SEED = 42

DATASETS = {
    "Датасет №1: Pima Indians Diabetes": "salihacur/diabetes",
    "Датасет №2: Churn Modelling": "shrutimechlearn/churn-modelling",
}

ALGORITHMS = ["K-Means", "DBSCAN", "Agglomerative"]
LINKAGES = ["ward", "complete", "average", "single"]

SPACE_PCA = "2 главные компоненты (PCA, как в ноутбуке)"
SPACE_FULL = "Все признаки (стандартизованные)"

# значения из ноутбука (кластеризация в пространстве двух главных компонент)
DEFAULT_K = {"df1": 3, "df2": 6}
DEFAULT_EPS_PCA = {"df1": 0.45, "df2": 0.2}
DEFAULT_EPS_FULL = 2.0


def read_csv_from_kaggle(slug):
    import kagglehub

    path = kagglehub.dataset_download(slug)
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    return pd.read_csv(os.path.join(path, csv_file))


@st.cache_resource(show_spinner="Загрузка и предобработка датасета...")
def prepare_dataset(ds_name):
    """Та же предобработка, что в ноутбуке: без целевой переменной, масштабирование, PCA."""
    df = read_csv_from_kaggle(DATASETS[ds_name])

    if ds_name.startswith("Датасет №1"):
        ds_key = "df1"
        zero_missing = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}
        raw = df.drop(columns="Outcome")
        cols = [c for c in raw.columns if c.lower() in zero_missing]
        raw[cols] = raw[cols].replace(0, np.nan)
        imputed = SimpleImputer(strategy="median").fit_transform(raw)
        features = pd.DataFrame(imputed, columns=raw.columns)
    else:
        ds_key = "df2"
        raw = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
        features = pd.get_dummies(raw, columns=["Geography", "Gender"],
                                  drop_first=True, dtype=int).reset_index(drop=True)

    X = StandardScaler().fit_transform(features)
    pca = PCA(n_components=3, random_state=SEED)
    coords = pca.fit_transform(X)

    return {
        "key": ds_key,
        "features": features,
        "X": X,
        "pca": coords,
        "var": pca.explained_variance_ratio_,
    }


def sample_idx(n_total, n_points):
    if n_points >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n_total, n_points, replace=False))


def get_space(data, space, idx):
    if space == SPACE_PCA:
        return data["pca"][idx, :2]
    return data["X"][idx]


@st.cache_data(show_spinner=False)
def run_clustering(ds_name, space, n_points, algo, params):
    data = prepare_dataset(ds_name)
    idx = sample_idx(len(data["X"]), n_points)
    Z = get_space(data, space, idx)
    extra = {}

    if algo == "K-Means":
        model = KMeans(n_clusters=params["k"], n_init=10, random_state=SEED).fit(Z)
        labels = model.labels_
        extra = {
            "inertia": float(model.inertia_),
            "inter": float(np.mean(pdist(model.cluster_centers_))),
        }
    elif algo == "DBSCAN":
        labels = DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit_predict(Z)
    else:
        labels = AgglomerativeClustering(n_clusters=params["k"], linkage=params["linkage"]).fit_predict(Z)

    # Silhouette считается без шумовых точек (-1) и только если кластеров не меньше двух
    mask = labels != -1
    n_valid = len(set(labels[mask]))
    if 1 < n_valid < mask.sum():
        sil = float(silhouette_score(Z[mask], labels[mask]))
    else:
        sil = np.nan
    return labels, sil, extra


@st.cache_data(show_spinner=False)
def elbow_curve(ds_name, space, n_points, k_max):
    data = prepare_dataset(ds_name)
    Z = get_space(data, space, sample_idx(len(data["X"]), n_points))
    ks = list(range(2, k_max + 1))
    inertia = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z).inertia_ for k in ks]
    return ks, inertia


@st.cache_data(show_spinner=False)
def k_distances(ds_name, space, n_points, min_samples):
    data = prepare_dataset(ds_name)
    Z = get_space(data, space, sample_idx(len(data["X"]), n_points))
    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)
    return np.sort(dist[:, min_samples - 1])


@st.cache_data(show_spinner=False)
def linkage_matrix(ds_name, space, n_points, linkage):
    data = prepare_dataset(ds_name)
    Z = get_space(data, space, sample_idx(len(data["X"]), n_points))
    return sch.linkage(Z, method=linkage)


def cluster_colors(labels):
    n = len(set(labels) - {-1})
    if n <= 10:
        palette = sns.color_palette("tab10", max(n, 1))
    elif n <= 20:
        palette = sns.color_palette("tab20", n)
    else:
        palette = sns.color_palette("husl", n)
    return palette


def draw_clusters(coords, labels, dim3, elev, azim, title, var):
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d") if dim3 else fig.add_subplot(111)
    n_dims = 3 if dim3 else 2
    palette = cluster_colors(labels)

    for lab in sorted(set(labels)):
        m = labels == lab
        pts = [coords[m, j] for j in range(n_dims)]
        if lab == -1:
            ax.scatter(*pts, c="lightgray", marker="x", s=20, label=f"Шум ({m.sum()})")
        else:
            ax.scatter(*pts, color=palette[lab % len(palette)], s=25, alpha=0.75,
                       label=f"Кластер {lab} ({m.sum()})")

    ax.set_xlabel(f"Главная компонента 1 ({var[0]:.0%})")
    ax.set_ylabel(f"Главная компонента 2 ({var[1]:.0%})")
    if dim3:
        ax.set_zlabel(f"Главная компонента 3 ({var[2]:.0%})")
        ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    if len(set(labels)) <= 15:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def params_to_text(d):
    return ", ".join(f"{k}={v}" for k, v in d.items())


def silhouette_comment(sil):
    if np.isnan(sil):
        return "Silhouette не определён: найдено меньше двух кластеров."
    if sil > 0.5:
        return "Silhouette выше 0.5 — кластеры выражены чётко."
    if sil >= 0.25:
        return "Silhouette 0.25–0.5 — структура слабая или умеренная."
    return "Silhouette ниже 0.25 — кластеры практически не выражены."


st.title("Лабораторная 6 — Кластеризация данных")
st.write(
    "Интерактивный визуализатор кластеров: выберите алгоритм, меняйте его параметры ползунками, "
    "и график кластеров и коэффициент силуэта пересчитаются сразу."
)

# --- боковая панель
st.sidebar.header("1. Данные")
ds_name = st.sidebar.selectbox("Выберите датасет", list(DATASETS))

try:
    data = prepare_dataset(ds_name)
except Exception as e:
    st.error(f"Не удалось загрузить датасет: {e}")
    st.stop()

ds_key = data["key"]
n_total = len(data["X"])

if n_total > 1000:
    n_points = st.sidebar.slider("Число объектов в выборке", 500, n_total, min(5000, n_total), step=500)
    if n_points > 5000:
        st.sidebar.caption("Agglomerative на большой выборке работает заметно медленнее.")
else:
    n_points = n_total
    st.sidebar.caption(f"Используются все {n_total} объектов.")

space = st.sidebar.radio("Пространство кластеризации", [SPACE_PCA, SPACE_FULL])
space_id = "pca" if space == SPACE_PCA else "full"

st.sidebar.header("2. Алгоритм")
algo = st.sidebar.selectbox("Алгоритм кластеризации", ALGORITHMS)

if algo == "K-Means":
    k = st.sidebar.slider("Количество кластеров k", 2, 12, DEFAULT_K[ds_key], key=f"km_k_{ds_key}")
    params = {"k": k}
elif algo == "DBSCAN":
    eps_default = DEFAULT_EPS_PCA[ds_key] if space == SPACE_PCA else DEFAULT_EPS_FULL
    eps = st.sidebar.slider("Радиус окрестности eps", 0.05, 5.0, eps_default, step=0.05,
                            key=f"db_eps_{ds_key}_{space_id}")
    min_samples = st.sidebar.slider("min_samples", 2, 30, 5, key=f"db_ms_{ds_key}")
    params = {"eps": round(eps, 2), "min_samples": min_samples}
else:
    k = st.sidebar.slider("Количество кластеров k", 2, 12, DEFAULT_K[ds_key], key=f"ag_k_{ds_key}")
    linkage = st.sidebar.selectbox("Метод объединения (linkage)", LINKAGES)
    params = {"k": k, "linkage": linkage}

st.sidebar.header("3. График")
dim = st.sidebar.radio("Размерность графика", ["2D", "3D"], horizontal=True)
elev, azim = 25, 45
if dim == "3D":
    elev = st.sidebar.slider("Угол наклона", 0, 90, 25)
    azim = st.sidebar.slider("Угол поворота", 0, 360, 45)

# --- кластеризация
with st.spinner(f"Кластеризация ({algo})..."):
    labels, sil, extra = run_clustering(ds_name, space, n_points, algo, params)

idx = sample_idx(n_total, n_points)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())

st.subheader(f"{algo} на «{ds_name}»")
st.caption(
    f"Объектов: {len(idx)}. Параметры: {params_to_text(params)}. Кластеризация в пространстве: {space.lower()}. "
    "График всегда строится в проекции на главные компоненты, поэтому при 3D третья ось — "
    "дополнительная проекция."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Кластеров", n_clusters)
c2.metric("Шумовых точек", n_noise)
c3.metric("Silhouette Score", "—" if np.isnan(sil) else f"{sil:.3f}")
c4.metric("Inertia (только K-Means)", f"{extra['inertia']:.1f}" if extra else "—")
if extra:
    st.caption(f"Среднее расстояние между центроидами: {extra['inter']:.2f}.")
st.caption(silhouette_comment(sil))

fig = draw_clusters(
    data["pca"][idx], labels, dim == "3D", elev, azim,
    f"{algo}: {params_to_text(params)}", data["var"],
)
st.pyplot(fig)
plt.close(fig)

# --- сравнение конфигураций
if "lab6_history" not in st.session_state:
    st.session_state["lab6_history"] = []

b1, b2, _ = st.columns([1, 1, 4])
if b1.button("Добавить в сравнение"):
    st.session_state["lab6_history"].append({
        "Датасет": ds_name.split(":")[1].strip(),
        "Пространство": "PCA 2D" if space == SPACE_PCA else "Все признаки",
        "Объектов": len(idx),
        "Алгоритм": algo,
        "Параметры": params_to_text(params),
        "Кластеров": n_clusters,
        "Шум": n_noise,
        "Silhouette": None if np.isnan(sil) else round(sil, 3),
        "Inertia": round(extra["inertia"], 1) if extra else None,
    })
if b2.button("Очистить сравнение"):
    st.session_state["lab6_history"] = []

if st.session_state["lab6_history"]:
    st.subheader("Сравнение конфигураций")
    st.dataframe(pd.DataFrame(st.session_state["lab6_history"]), use_container_width=True, hide_index=True)

# --- профили кластеров
with st.expander("Профили кластеров (средние значения исходных признаков)"):
    feats = data["features"].iloc[idx].reset_index(drop=True).copy()
    feats["Кластер"] = np.where(labels == -1, "Шум", labels.astype(str))
    profile = feats.groupby("Кластер").mean()
    profile.insert(0, "Размер", feats.groupby("Кластер").size())
    st.dataframe(profile.round(2), use_container_width=True)

# --- вспомогательные графики для подбора параметров
with st.expander("Подбор параметров (метод локтя / k-расстояния / дендрограмма)"):
    if algo == "K-Means":
        ks, inertia = elbow_curve(ds_name, space, n_points, 12)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, inertia, marker="o")
        ax.axvline(params["k"], color="red", linestyle="--", label=f"выбранное k={params['k']}")
        ax.set_xlabel("Количество кластеров (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Метод локтя (Elbow Method)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    elif algo == "DBSCAN":
        dist = k_distances(ds_name, space, n_points, params["min_samples"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dist)
        ax.axhline(params["eps"], color="red", linestyle="--", label=f"выбранное eps={params['eps']}")
        ax.set_xlabel("Точки, отсортированные по расстоянию")
        ax.set_ylabel(f"{params['min_samples']}-е расстояние до ближайшего соседа")
        ax.set_title("График k-расстояний")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        link = linkage_matrix(ds_name, space, n_points, params["linkage"])
        fig, ax = plt.subplots(figsize=(10, 4))
        sch.dendrogram(link, truncate_mode="lastp", p=30, show_leaf_counts=True, no_labels=True, ax=ax)
        ax.set_xlabel("Объединённые группы объектов")
        ax.set_ylabel("Расстояние (Distance)")
        ax.set_title(f"Дендрограмма (linkage={params['linkage']}, показаны последние 30 объединений)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
