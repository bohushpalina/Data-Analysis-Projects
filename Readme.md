# Data-Analysis-Projects

A collection of data analysis and machine learning laboratory projects implemented in Python using Jupyter Notebook.
The repository covers data exploration, visualization, statistical analysis, linear regression, and classification.
Labs 2–4 also include optional interactive pages built with **Streamlit**, combined into **one multipage web app**.

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

## Interactive Web App

All optional Streamlit tasks of Labs 2–4 are combined on **one website**.

**Open the app:** `<link to the deployed app>`

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

> The first load of Labs 2 and 4 can take a little longer while the datasets are downloaded.

## Technologies

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
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
│   └── 3_Lab4_Classification.py
├── lab1_exp_eval.ipynb
├── lab2_exploratory_analysis.ipynb
├── lab3_linear_regression.ipynb
├── lab4_classification_analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Author

**Palina Bohush**
