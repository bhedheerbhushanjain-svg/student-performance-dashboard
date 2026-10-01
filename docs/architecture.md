# System Architecture & Component Design

## Architectural Overview

The **Student Performance Analysis Dashboard** is designed using a modular, decoupled architecture following software engineering best practices. The codebase separates data ingestion, feature transformation, analytical computation, visualization rendering, and web presentation.

---

## High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                        WEB PRESENTATION LAYER                          │
│                                                                        │
│   Streamlit Web Interface (dashboard/app.py)                           │
│   ├─ Sidebar Controls & Multi-Select Filters                           │
│   ├─ Real-Time KPI Cards                                               │
│   ├─ 6 Interactive Thematic Tabs                                       │
│   └─ Interactive Early Warning Grade Predictor                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Invokes
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       APPLICATION & LOGIC LAYER                        │
│                                                                        │
│   ┌──────────────────────────────┐    ┌─────────────────────────────┐  │
│   │   Statistical & ML Core      │    │    Visualization Engine     │  │
│   │   (src/analysis.py)          │    │    (src/visualizations.py)  │  │
│   │   • Descriptive Stats / IQR  │    │    • 12 Plotly Interactive  │  │
│   │   • Pearson Correlation      │    │      Figures                │  │
│   │   • OLS Linear Regression    │    │    • Publication Themes     │  │
│   │   • Random Forest Regressor  │    │    • Responsive Box/Violin  │  │
│   │   • Automated Insights       │    │    • Heatmap & Subplots     │  │
│   └──────────────▲───────────────┘    └──────────────▲──────────────┘  │
│                  │                                   │                 │
│                  └─────────────────┬─────────────────┘                 │
│                                    │ Consumes Clean Matrices           │
│                                    ▼                                   │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │               Preprocessing & Feature Pipeline                  │  │
│   │               (src/preprocessing.py)                            │  │
│   │   • Label Decoders (LABEL_MAPPINGS)                             │  │
│   │   • Missing Value Imputation (Median / Mode)                    │  │
│   │   • One-Hot Categorical Encoding                                │  │
│   │   • Regime Partitioning (Regime A: Leakage / Regime B: Clean)   │  │
│   └────────────────────────────────▲────────────────────────────────┘  │
│                                    │ Loads Raw Records                 │
│                                    ▼                                   │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     Data Loader Pipeline                        │  │
│   │                     (src/data_loader.py)                        │  │
│   │   • Safe File Resolution (resolve_data_path)                    │  │
│   │   • Semicolon CSV Parser                                        │  │
│   │   • Schema & Boundary Validation                                │  │
│   │   • Multi-Subject & Merged Cohort Reassembly                    │  │
│   └─────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Reads
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         DATA STORAGE LAYER                             │
│                                                                        │
│   data/raw/                                                            │
│   ├── student-mat.csv   (Mathematics Course - 395 records)             │
│   ├── student-por.csv   (Portuguese Course  - 649 records)             │
│   ├── student-merge.R   (Cortez & Silva R merge script)                │
│   └── student.txt       (Attribute documentation)                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Component Roles & Responsibilities

### 1. Ingestion Layer (`src/data_loader.py`)
- Provides deterministic loading for single subject (`mat` or `por`), combined (`1,044` rows), and linked cohorts (`382` rows).
- Enforces strict boundary checks on grades $[0, 20]$ and age $[15, 25]$ through `validate_dataset()`.

### 2. Feature Pipeline (`src/preprocessing.py`)
- Separates presentation data from modeling data.
- Supplies human-readable labels (`label_studytime`, `label_sex`, etc.) for chart readability.
- Implements two isolated feature preparation paths:
  * `prepare_ml_features(include_prior_grades=True)`: Regime A.
  * `prepare_ml_features(include_prior_grades=False)`: Regime B.

### 3. Analytical Engine (`src/analysis.py`)
- Calculates mathematical statistics: sample mean, median, standard deviation, quartiles, IQR, skewness, kurtosis.
- Trains `LinearRegression` and `RandomForestRegressor` models using scikit-learn.
- Generates dynamic, empirical insight statements directly from the active dataset.

### 4. Visualization Engine (`src/visualizations.py`)
- Encapsulates all plotting logic using Plotly Graph Objects and Express.
- Ensures zero coupling between raw chart rendering code and web layout logic.

### 5. Presentation Layer (`dashboard/app.py`)
- Manages application state, sidebar filter interactions, metric card calculations, and multi-tab rendering.
- Uses Streamlit caching (`@st.cache_data`) for instantaneous reload performance.

---

## Container Architecture (Docker)

```
┌────────────────────────────────────────────────────────────────┐
│             Docker Container: student-performance-dashboard    │
│                                                                │
│  Base: python:3.11-slim (Debian Linux)                         │
│  User: appuser (UID: 1000, Non-Root)                           │
│  Port: 8501 (Streamlit Web Port)                               │
│  Healthcheck: curl -f http://localhost:8501/_stcore/health     │
│  Volumes: /app (Codebase + Raw Benchmark Data)                 │
└────────────────────────────────────────────────────────────────┘
```
