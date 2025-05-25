# Regression Model Comparison on Diabetes Dataset 

This project systematically compares the performance of ten different regression models on the Diabetes dataset. The primary objective is to identify which models best predict blood glucose levels based on various clinical measurements.

## Project Objective

To evaluate and compare ten regression models for predicting plasma glucose concentration in diabetes patients, utilizing consistent preprocessing, training, and validation procedures.

---

## Dataset Overview

The dataset consists of clinical measurements from diabetes patients.

* **Source File:** `diabetes.csv`
* **Features Used for Prediction (X):**
    * Pregnancies: Number of times pregnant
    * BloodPressure: Diastolic blood pressure (mm Hg)
    * SkinThickness: Triceps skin fold thickness (mm)
    * Insulin: 2-Hour serum insulin (mu U/ml)
    * BMI: Body mass index (weight in kg/(height in m)^2)
    * DiabetesPedigreeFunction: Diabetes pedigree function
    * Age: Age of the patient (years)
    * Outcome: Whether the patient has diabetes (binary, though not used as a feature for glucose prediction in this regression task after initial feature selection steps in some parts of the notebook).
* **Target Variable (y):**
    * Glucose: Plasma glucose concentration (mg/dL)

---

## Project Workflow

1.  **Data Loading & Initial Analysis:**
    * Loaded the dataset from `diabetes.csv`.
    * Analyzed feature correlations using a heatmap.
    * Visualized feature distributions using histograms.

2.  **Data Preprocessing:**
    * **Handling Missing Values (Implicit):** Columns 'BloodPressure', 'SkinThickness', 'Insulin', and 'BMI' had zero values replaced with NaN, as zero is not a physiologically plausible value for these measurements.
    * **Outlier Detection & Removal:** Outliers were removed using the Interquartile Range (IQR) method (1.5 * IQR threshold).
    * **Train-Test Split:** Data was split into training (80%) and testing (20%) sets.
    * **Feature Scaling:** Numerical features were scaled using `StandardScaler`.
    * **Missing Value Imputation:** `KNNImputer` (with n_neighbors=5) was used to fill NaN values after scaling.

3.  **Feature Engineering & Selection:**
    * **Feature Selection:** `SelectKBest` with `f_regression` was applied to select the top 4 features from the imputed training data (identified in the notebook as 'BloodPressure', 'Insulin', 'Age', 'Outcome').
    * **Dimensionality Reduction:** Principal Component Analysis (PCA) was applied to the 4 selected features, resulting in 4 principal components.
    * **Polynomial Features:** Second-degree polynomial features were generated from the PCA output for the Polynomial Regression model.

4.  **Model Training & Cross-Validation:**
    * Ten regression models were defined and evaluated:
        1.  Linear Regression
        2.  Polynomial Regression (using polynomial features from PCA output)
        3.  K-Nearest Neighbors (KNN) Regression
        4.  Support Vector Regression (SVR)
        5.  Decision Tree Regression
        6.  Random Forest Regression
        7.  Ridge Regression
        8.  Lasso Regression
        9.  Bayesian Linear Regression
        *All models (except Polynomial Regression) were cross-validated using the PCA-transformed selected features.*
    * **Cross-Validation Strategy:** 5-fold cross-validation.
    * **Evaluation Metrics:** Mean Squared Error (MSE) and R² score.
    * **Cross-Validation Findings:** Linear Regression, Ridge Regression, and Bayesian Linear Regression demonstrated the best and most stable performance (CV MSE ~535-536, CV R² ~0.351). Ridge Regression was chosen for further tuning due to its regularization capabilities.

5.  **Hyperparameter Tuning:**
    * `GridSearchCV` was used to tune the `alpha` parameter for the Ridge Regression model on the PCA-transformed training data.
    * **Best Alpha:** 0.0001.
    * **Best Cross-Validated MSE (Negative):** 535.40.

6.  **Final Model Evaluation:**
    * The best Ridge Regression model (with alpha=0.0001) was fitted on the PCA-transformed selected training features and evaluated on the corresponding test set.
    * **Test MSE:** 549.44
    * **Test R²:** 0.366

---

## Technologies & Libraries Used

* **Python 3**
* **Core Libraries:**
    * Pandas
    * NumPy
    * Matplotlib
    * Seaborn
* **Scikit-learn:**
    * `train_test_split`, `cross_val_score`, `GridSearchCV`
    * `LinearRegression`, `Ridge`, `Lasso`, `BayesianRidge`
    * `PolynomialFeatures`, `StandardScaler`
    * `KNeighborsRegressor`
    * `SVR` (Support Vector Regressor)
    * `DecisionTreeRegressor`
    * `RandomForestRegressor`
    * `mean_squared_error`, `r2_score`
    * `SelectKBest`, `f_regression`
    * `KNNImputer`
    * `PCA` (Principal Component Analysis)

