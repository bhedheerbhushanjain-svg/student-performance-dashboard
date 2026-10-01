# Mathematical & Analytical Methodology

## Overview

This document details the analytical lifecycle implemented in the **Student Performance Analysis Dashboard**, covering data ingestion, preprocessing, exploratory data analysis, statistical metrics, and predictive machine learning modeling.

---

## 1. Data Ingestion & Integrity Verification

1. **Format Handling:** Raw records are stored in semicolon-separated CSV files (`student-mat.csv`, `student-por.csv`).
2. **Schema Ingestion:** Ingestion routines in `src/data_loader.py` enforce column validation against `EXPECTED_COLUMNS`.
3. **Boundary Verification:**
   - Age is validated within $[15, 25]$.
   - Grade variables ($G1, G2, G3$) are validated within $[0, 20]$.
   - Study time is validated within $[1, 4]$.
4. **Cohort Reconstruction:** Linked cohort reconstruction reproduces the 382 common students across courses via inner join on 13 demographic keys.

---

## 2. Preprocessing & Feature Engineering

1. **Missing Value Resilience:** While the official benchmark dataset contains zero missing values, `src/preprocessing.py` provides deterministic fallback imputation:
   $$\hat{x}_{\text{num}} = \text{median}(X_{\text{col}})$$
   $$\hat{x}_{\text{cat}} = \text{mode}(X_{\text{col}})$$
2. **Binary Transformation:** Attributes (`schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`) are mapped to binary integers $\{0, 1\}$.
3. **Nominal Categorical Encoding:** One-hot encoding with first-category removal ($\text{drop\_first}=\text{True}$) prevents the dummy variable trap in linear models.
4. **Label Mapping:** Separate human-readable label columns (e.g. `label_studytime`, `label_failures`) are attached for presentation without contaminating numeric arrays.

---

## 3. Statistical Analysis & Distribution Metrics

### Descriptive Statistics
For grade distributions and numerical features, the pipeline computes both parametric and non-parametric statistics:
- **Sample Mean ($\bar{x}$):**
  $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
- **Sample Standard Deviation ($s$):**
  $$s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}$$
- **Median & Quartiles:** $Q_1$ (25th percentile), $Q_2$ (50th percentile / median), $Q_3$ (75th percentile).
- **Interquartile Range (IQR):**
  $$\text{IQR} = Q_3 - Q_1$$
- **Sample Skewness ($g_1$):** Measures distributional asymmetry:
  $$g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3$$
- **Sample Kurtosis ($g_2$):** Measures tail heaviness:
  $$g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$

### Pearson Correlation Coefficient
Linear relationships between feature $X$ and final grade $Y = G3$ are evaluated via:
$$r_{X, Y} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

---

## 4. Machine Learning & Predictive Modeling

### Model Architectures
1. **Ordinary Least Squares (OLS) Linear Regression:**
   $$\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j X_j$$
   Coefficients $\boldsymbol{\beta}$ are estimated by minimizing residual sum of squares:
   $$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$$
2. **Random Forest Regressor:**
   An ensemble of $B = 100$ decorrelated regression decision trees trained on bootstrap samples of the training data:
   $$\hat{y}_{\text{RF}} = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$

### Evaluation Metrics
Performance is calculated on an unseen $20\%$ test partition ($\text{random\_state}=42$):
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

---

## 5. Experimental Regimes & Data Leakage Proof

| Metric | Regime A: Linear Regression (With G1/G2) | Regime A: Random Forest (With G1/G2) | Regime B: Linear Regression (Without G1/G2) | Regime B: Random Forest (Without G1/G2) |
|---|---|---|---|---|
| **$R^2$ Score** | **$0.7241$** | **$0.8161$** | **$0.1415$** | **$0.2682$** |
| **RMSE** | **$2.0945$** | **$1.7088$** | **$3.6934$** | **$3.4093$** |
| **MAE** | **$1.6467$** | **$1.1643$** | **$3.3953$** | **$3.1064$** |
| **Features** | 41 features | 41 features | 39 features | 39 features |

### Empirical Interpretation:
The massive drop in $R^2$ from $0.816$ to $0.268$ demonstrates that the high predictive accuracy of Regime A is driven almost entirely by the high collinearity between prior exam results and final grades. In an educational early-warning scenario where intervention must happen *prior* to examinations, Regime B is the only scientifically valid configuration.
