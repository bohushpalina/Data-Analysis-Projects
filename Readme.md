# Data-Analysis-Projects

A collection of data analysis and machine learning laboratory projects implemented in Python using Jupyter Notebook.
The repository covers data exploration, visualization, statistical analysis, linear regression, classification, ensemble methods, and clustering.
Labs 2–6 also include optional interactive pages built with **Streamlit**, combined into **one multipage web app**.

## Labs

### Lab 1 — Basic Data Analysis

Introduction to working with datasets using **Pandas** and **Matplotlib**.

The lab uses the **Iris** and **Titanic** datasets to load data, inspect their structure, and visualize selected numerical features.

**Topics:**

* Loading datasets with Pandas
* Basic data inspection
* Data visualization with Matplotlib
* Working with numerical features

**Notebook:** `lab1_exp_eval.ipynb`

---

### Lab 2 — Exploratory Data Analysis

Exploratory analysis of **Math Students Performance** and **Billionaires Statistics** datasets.

The lab focuses on dataset structure, descriptive statistics, missing values, duplicates, distributions, outliers, and relationships between numerical variables.

**Topics:**

* Data loading and preprocessing
* Descriptive statistics
* Missing values and duplicates
* Histograms, KDE plots, boxplots, and violin plots
* Correlation analysis
* Heatmaps and scatter plots
* Data interpretation

**Notebook:** `lab2_exploratory_analysis.ipynb`

**Streamlit page (optional task, Part 3):** `pages/1_Lab2_EDA.py` — interactive EDA dashboard. The key elements of graphical analysis and data cleaning are moved into the web interface:

* dataset selection (`st.selectbox()`)
* data cleaning controls (drop duplicates, handle missing values)
* filters (`st.multiselect()`, `st.slider()`) with a dynamically updated table (`st.dataframe()`)
* choice of chart type (histogram, KDE, boxplot, violin plot, correlation heatmap, scatter plot)

---

### Lab 3 — Linear Regression

Application of linear regression models to real-world datasets.

The lab includes simple and multiple linear regression, feature selection, and model evaluation using standard regression metrics.

**Topics:**

* Train/test splitting
* Simple linear regression
* Multiple linear regression
* Feature selection
* Model evaluation with MAE, MSE, and R²
* Data visualization
* Saving trained models with Joblib

**Notebook:** `lab3_linear_regression.ipynb`

**Streamlit page (optional task, Part 3):** `pages/2_Lab3_Regression.py` — interactive prediction simulator:

* sliders (`st.slider()`) or number inputs (`st.number_input()`) in the sidebar for manual entry of the independent variables
* the model trained in the notebook is exported with Joblib (`saved_models/`) and loaded in the app
* the forecast of the target variable is recalculated instantly with every change of the controls
* a live chart of predictions that updates in real time

> Run the Lab 3 notebook first so that the saved models appear in `saved_models/`.

---

### Lab 4 — Classification

Application of classification algorithms to binary classification problems on two real-world datasets: **Pima Indians Diabetes** (medical diagnosis) and **Churn Modelling** (bank customer churn).

The lab includes data preprocessing, training of four classifiers with default parameters, hyperparameter tuning with cross-validation, and comparison of the models using classification metrics.

**Topics:**

* Data preprocessing (missing values, categorical encoding, scaling with `StandardScaler`)
* Stratified train/test splitting
* k-Nearest Neighbors, Decision Tree, SVM, and MLP classifiers
* Hyperparameter tuning with `GridSearchCV` (5-fold cross-validation)
* Visualization of decision boundaries, decision tree, and neural network architecture
* Model evaluation with Accuracy, Precision, Recall, and F1-score
* Confusion matrices and comparison of models on imbalanced classes

**Notebook:** `lab4_classification_analysis.ipynb`

**Streamlit page (optional task, Part 4):** `pages/3_Lab4_Classification.py` — classification web interface:

* classifier selection with a drop-down list (`st.selectbox()`): k-NN, Decision Tree, SVM, MLP
* main hyperparameters of the chosen model are changed directly on the page (e.g. `n_neighbors`, `max_depth`, `C`, `kernel`, `hidden_layer_sizes`) using `st.slider()` / `st.selectbox()`
* the model is retrained on every change, and the confusion matrix and the metrics table (Accuracy, Precision, Recall, F1-score) are updated in real time

---

### Lab 5 — Ensemble Methods

Application of ensemble methods to the same two classification datasets as in Lab 4: **Pima Indians Diabetes** and **Churn Modelling**. The preprocessing and the stratified 80/20 train/test split are the same as in Lab 4, so the ensembles can be compared fairly with the classical models.

The lab includes bagging, two gradient boosting libraries, stacking, tuning of the key parameters with cross-validation, and a summary comparison of classical models and ensembles.

**Topics:**

* Bagging (`BaggingClassifier`), number of estimators selected by cross-validation (`n_estimators = 75` for both datasets)
* Gradient boosting with specialized libraries instead of scikit-learn: **LightGBM** (`LGBMClassifier`) and **XGBoost** (`XGBClassifier`)
* Tuning of `n_estimators`, `learning_rate`, and `max_depth` with a universal `cross_valid` function (5-fold cross-validation)
* Stacking (`StackingClassifier`): SVM and MLP as base models, Logistic Regression as the meta-model (its `C` is tuned)
* Confusion matrices and classification reports for every ensemble
* Summary table comparing the best classical models from Lab 4 with the ensembles

**Best parameters:**

| Model | Diabetes | Churn |
|---|---|---|
| Bagging | `n_estimators=75` | `n_estimators=75` |
| LightGBM | `n_estimators=50`, `learning_rate=0.03`, `max_depth=4` | `n_estimators=50`, `learning_rate=0.05`, `max_depth=5` |
| XGBoost | `n_estimators=300`, `learning_rate=0.05`, `max_depth=2` | `n_estimators=50`, `learning_rate=0.05`, `max_depth=5` |
| Stacking | meta-model `C=1` | meta-model `C=0.1` |

**Results on the test set (sorted by F1):**

*Diabetes*

| Model | Type | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LightGBM | Ensemble | 0.766 | 0.705 | 0.574 | 0.633 |
| XGBoost | Ensemble | 0.753 | 0.674 | 0.574 | 0.620 |
| kNN | Classic | 0.734 | 0.633 | 0.574 | 0.602 |
| MLP | Classic | 0.727 | 0.625 | 0.556 | 0.588 |
| Bagging | Ensemble | 0.727 | 0.630 | 0.537 | 0.580 |
| Stacking | Ensemble | 0.721 | 0.622 | 0.519 | 0.566 |
| SVM | Classic | 0.708 | 0.605 | 0.481 | 0.536 |
| Decision Tree | Classic | 0.688 | 0.636 | 0.259 | 0.368 |

*Churn*

| Model | Type | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| MLP | Classic | 0.861 | 0.717 | 0.523 | 0.605 |
| Bagging | Ensemble | 0.856 | 0.714 | 0.491 | 0.582 |
| Stacking | Ensemble | 0.865 | 0.787 | 0.462 | 0.582 |
| XGBoost | Ensemble | 0.868 | 0.833 | 0.440 | 0.576 |
| LightGBM | Ensemble | 0.864 | 0.812 | 0.435 | 0.566 |
| SVM | Classic | 0.861 | 0.834 | 0.396 | 0.537 |
| Decision Tree | Classic | 0.856 | 0.787 | 0.400 | 0.531 |
| kNN | Classic | 0.833 | 0.695 | 0.319 | 0.438 |

On Diabetes the boosting models (LightGBM, XGBoost) give the best F1, ahead of the best classical model (kNN). On Churn the ensembles reach the highest Accuracy (XGBoost: 0.868) and a high Precision (XGBoost: 0.833, SVM has 0.834), but the classical MLP keeps the best F1, because Recall of the minority class (about 20% of clients leave) stays below 0.53 for all models. Precision, Recall, and F1 are calculated for class `1`.

**Notebook:** `lab5_ensembles.ipynb`

**Streamlit page (optional task, Part 4):** `pages/4_Lab5_ensembles.py` — "Classic vs Ensembles" benchmark:

* the test subset is the 20% test sample from the notebook (its share can be reduced with a slider and downloaded as CSV) or your own uploaded CSV with the same columns
* models for the current comparison are chosen with checkboxes (`st.checkbox()`): kNN, Decision Tree, SVM, MLP, Bagging, LightGBM, XGBoost, Stacking
* ROC curves, bar charts of metrics (Accuracy, Precision, Recall, F1, ROC-AUC), or both are drawn for all selected models on one canvas
* a metrics table, confusion matrices, and the hyperparameters of the selected models

> The summary table in the notebook reads the Lab 4 results from `lab4_results_df1.csv` and `lab4_results_df2.csv`, so these files must be next to the notebook. The page needs the `xgboost` and `lightgbm` packages.

---

### Lab 6 — Clustering

Application of unsupervised learning to the **Pima Indians Diabetes** (768 patients) and **Bank Customer Churn** (10,000 clients) datasets. The target variables (`Outcome`, `Exited`) are removed, so the data is clustered without labels.

The lab includes preprocessing (median imputation of medical zeros, one-hot encoding, `StandardScaler`), dimensionality reduction with PCA to 2 components, three clustering algorithms based on different approaches, parameter selection, and internal validation of cluster quality.

**Topics:**

* Preprocessing and scaling of features, PCA for 2D visualization
* K-Means (partitioning approach) with the Elbow Method: k = 3 for Diabetes, k = 6 for Churn
* DBSCAN (density-based approach) with a k-distance graph for choosing `eps` (`min_samples = 5`)
* Agglomerative Clustering (hierarchical approach, Ward linkage) and dendrograms
* Scatter plots of clusters for each algorithm and dataset
* Validation with Inertia, mean distance between centroids, and Silhouette Score
* Summary comparison table for all algorithms on both datasets

**Results (on the 2D PCA projection):**

| Dataset | Algorithm | Inertia | Mean distance between centroids | Silhouette |
|---|---|---|---|---|
| Diabetes | K-Means (k=3) | 1110.76 | 2.68 | 0.409 |
| Diabetes | DBSCAN (eps=0.45) | – | – | n/a (1 cluster, 39 noise points) |
| Diabetes | Agglomerative (k=3) | – | – | 0.367 |
| Churn | K-Means (k=6) | 4683.91 | 2.39 | 0.416 |
| Churn | DBSCAN (eps=0.2) | – | – | 0.297 (4 clusters, 63 noise points) |
| Churn | Agglomerative (k=6) | – | – | 0.382 |

K-Means gives the best Silhouette Score on both datasets, but all values stay below about 0.42, which means moderately expressed cluster structure. Inertia and the distance between centroids are only defined for K-Means and cannot be compared across datasets. The notebook ends with a short explanation of each metric and of the results.

**Notebook:** `lab6_clustering.ipynb`

**Streamlit page (optional task, Part 4):** `pages/5_Lab6_Clustering.py` — interactive cluster visualizer:

* clustering algorithm selection with a drop-down list (`st.selectbox()`): K-Means, DBSCAN, Agglomerative
* sliders (`st.slider()`) for the parameters: `k` for K-Means and Agglomerative, `eps` and `min_samples` for DBSCAN
* the 2D/3D cluster plot is rebuilt and the Silhouette Score is recalculated on every change
* extras: choice of the clustering space (2 PCA components or all features), sample size, a table for comparing saved configurations, cluster profiles, and helper plots (elbow method, k-distance graph, dendrogram)

---

## Interactive Web App

All optional Streamlit tasks of Labs 2–6 are combined on **one website**.

**Open the app:** `http://localhost:8501`

Use the menu on the left to switch between the labs. All settings are in the sidebar or on the page itself, and the results update instantly — no code needed.

### Lab 2 — EDA Dashboard

1. Choose a dataset (Math Students or Billionaires).
2. Clean the data: remove duplicates, choose how to handle missing values.
3. Apply filters by numeric ranges or categories.
4. Browse the table, descriptive statistics, and missing values in the tabs.
5. In the **Charts** tab, choose a chart type (histogram, KDE, boxplot, violin, bar, pie, scatter, heatmap) and its settings.

### Lab 3 — Prediction Simulator

1. Choose a trained model (simple or multiple regression).
2. Set the feature values with sliders (or number fields) in the sidebar, for example advertising budgets for TV, Radio and Newspaper.
3. The forecast of the target variable (e.g. sales) is recalculated instantly and shown together with the regression formula.
4. The charts update in real time: the regression line with your point and the contribution of each feature.

### Lab 4 — Classification

1. Choose a dataset (Diabetes or Churn).
2. Choose a classifier from the drop-down list (kNN, Decision Tree, SVM, MLP).
3. Change its hyperparameters in the sidebar (e.g. number of neighbors, tree depth, `C`, number of neurons).
4. The confusion matrix and the metrics table (Accuracy, Precision, Recall, F1-score) update automatically.
5. Press **Add to comparison** to save a configuration and compare several settings in one table.

### Lab 5 — Classic vs Ensembles

1. Choose a dataset (Diabetes or Churn).
2. Choose the test subset: the test sample from the notebook (with a slider for its share) or your own CSV file.
3. Tick the classical and ensemble models you want to compare.
4. Choose what to show: ROC curves, bar charts of metrics, or both.
5. The metrics table, charts, and confusion matrices update automatically.

### Lab 6 — Clustering Visualizer

1. Choose a dataset (Diabetes or Churn) and, for the large Churn dataset, the sample size.
2. Choose the clustering space: two PCA components (as in the notebook) or all standardized features.
3. Choose an algorithm from the drop-down list (K-Means, DBSCAN, Agglomerative).
4. Change its parameters with sliders (`k`, or `eps` and `min_samples`).
5. The 2D/3D plot and the Silhouette Score update automatically; in 3D the view angle can be rotated with sliders.
6. Press **Add to comparison** to save a configuration and compare several settings in one table.

> The first load of Labs 2, 4, 5 and 6 can take a little longer while the datasets are downloaded.

## Technologies

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* XGBoost
* LightGBM
* Joblib
* Streamlit
* KaggleHub

## Repository Structure

```text
Data-Analysis-Projects/
├── datasets/
│   └── Dataset files used in the laboratory projects
├── saved_models/
│   └── Saved machine learning models
├── pages/
│   ├── 1_Lab2_EDA.py
│   ├── 2_Lab3_Regression.py
│   ├── 3_Lab4_Classification.py
│   ├── 4_Lab5_ensembles.py
│   └── 5_Lab6_Clustering.py
├── lab1_exp_eval.ipynb
├── lab2_exploratory_analysis.ipynb
├── lab3_linear_regression.ipynb
├── lab4_classification_analysis.ipynb
├── lab5_ensembles.ipynb
├── lab6_clustering.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Author

**Palina Bohush**
