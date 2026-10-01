# Student Performance Analysis Dashboard

[![Automated Test Suite](https://github.com/bhedheerbhushanjain-svg/student-performance-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/bhedheerbhushanjain-svg/student-performance-dashboard/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.64-FF4B4B.svg)](https://streamlit.io/)
[![Docker: Ready](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](Dockerfile)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

An academic **Open Source Technologies (OST)** project delivering an end-to-end data analytics, interactive visualization, and machine learning dashboard for secondary education academic performance.

Developed strictly in accordance with modern open-source engineering workflows: GitFlow branching, Conventional Commits, Pytest validation, GitHub Actions multi-OS continuous integration, Docker containerization, and rigorous scientific ethics addressing **temporal data leakage**.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Dataset Provenance](#dataset-provenance)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Data Analysis](#data-analysis)
- [Visualisations](#visualisations)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Results](#results)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Running with Streamlit](#running-with-streamlit)
- [Running with Docker](#running-with-docker)
- [Git Workflow](#git-workflow)
- [Linux/Git Bash Commands](#linuxgit-bash-commands)
- [Testing](#testing)
- [GitHub Actions](#github-actions)
- [Limitations](#limitations)
- [Ethical/Data Considerations](#ethicaldata-considerations)
- [License](#license)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [Author](#author)

---

## Overview

Educational institutions face complex challenges in early identification of students at risk of academic failure. The **Student Performance Analysis Dashboard** ingests empirical educational data from the renowned UCI Student Performance benchmark, applies clean data engineering and exploratory data analysis, and presents key patterns via an interactive **Streamlit** web application. 

Crucially, this project moves beyond standard toy notebooks by demonstrating real open-source practices: clean separation of concerns, containerization, automated testing, version control governance, and honest scientific reporting that exposes rather than conceals machine learning data leakage.

---

## Problem Statement

Educational achievement is influenced by a web of demographic, social, familial, and behavioral factors:
1. How significantly do non-academic variables (e.g., weekly study hours, parental education levels, home internet access, school absences) correlate with secondary school outcomes?
2. Why do many machine learning student prediction models reported in academic literature claim artificially high accuracy ($R^2 > 0.80$), and how does the inclusion of intermediate period exams ($G1$, $G2$) introduce severe temporal data leakage?
3. How can educators build an actionable **early-warning system** that diagnoses risk *before* the semester exams take place?

---

## Objectives

- **OST Workflow Demonstration:** Implement professional Git version control, GitFlow branching (`main`, `develop`, `feature/*`), Conventional Commits, GitHub Actions CI, and Docker containerization.
- **Reproducible Data Pipeline:** Build modular ingestion and preprocessing routines validating schema constraints and handling data types safely.
- **Interactive Web Interface:** Provide a responsive Streamlit dashboard featuring multi-criteria sidebar filters, metric cards, and 12+ publication-quality Plotly charts.
- **Statistical Rigor:** Compute complete parametric and non-parametric statistical metrics (mean, median, standard deviation, quartiles, IQR, skewness, kurtosis).
- **Data Leakage Exploration:** Empirically compare predictive models trained **with** intermediate grades (Regime A) versus **without** intermediate grades (Regime B).
- **Automated Insights:** Generate dynamic, mathematically computed observations directly from active data subsets without hardcoded assumptions.

---

## Dataset

The project utilizes the authentic, publicly accessible **UCI Student Performance Dataset**, collected from two public secondary schools in the Alentejo region of Portugal (Gabriel Pereira and Mousinho da Silveira) during the 2005–2006 academic year.

- **Mathematics Cohort (`student-mat.csv`):** 395 student instances, 33 attributes.
- **Portuguese Language Cohort (`student-por.csv`):** 649 student instances, 33 attributes.
- **Merged Cohort (`student-merge.R`):** 382 distinct students enrolled in both subjects, linked via 13 common demographic identifiers.
- **Target Variable:** `G3` (Final academic period grade, scored on a scale from 0 to 20; pass threshold = 10).

---

## Dataset Provenance

- **Original Research Authors:** Paulo Cortez and Alice Silva (Department of Information Systems, University of Minho, Guimarães, Portugal).
- **Primary Source:** [UCI Machine Learning Repository - Student Performance (ID: 320)](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Secondary Access / Mirror Source:** [Kaggle Student Performance Data Set](https://www.kaggle.com/dskagglemt/student-performance-data-set/metadata)
- **Academic Citation:**
  > Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito & J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, EUROSIS, ISBN 978-9077381-39-7.
- **Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## Features

1. **Cohort Selection:** Seamlessly switch between Mathematics (395), Portuguese (649), Combined (1,044), and Matched Overlapping (382) student cohorts.
2. **Dynamic Multi-Select Filtering:** Real-time filtering by School, Gender, Age Range, Study Time, Past Failures, and Home Internet Connectivity.
3. **Live KPI Metric Cards:** Real-time feedback displaying cohort size, mean final grade, median grade, passing rate percentage ($\ge 10/20$), and dropout/zero-score count.
4. **Interactive Exploratory Charts:** Rich Plotly charts with custom color palettes, annotations, and hover details.
5. **Statistical Deep-Dive:** Full descriptive summary table and interactive Pearson correlation heatmap.
6. **Machine Learning & Data Leakage Laboratory:** Side-by-side empirical performance comparison (Regime A vs Regime B) measuring MAE, MSE, RMSE, and $R^2$.
7. **Interactive Early-Warning Predictor:** Sandbox allowing users to adjust student background attributes and generate live predictions from the clean model.
8. **Automated Insights Generator:** Dynamically derived analytical takeaways calculated directly from the filtered cohort.

---

## Technologies Used

- **Language:** Python 3.10, 3.11, 3.12
- **Data Engineering:** Pandas, NumPy
- **Visualizations:** Plotly Express & Graph Objects, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Linear Regression, Random Forest Regressor)
- **Web Application Framework:** Streamlit
- **Testing & Quality Assurance:** Pytest
- **Version Control:** Git, GitHub, GitFlow branching
- **Continuous Integration (CI):** GitHub Actions
- **Containerization & Deployment:** Docker, Docker Compose
- **Build Automation:** GNU Make / Makefile

---

## Project Architecture

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
│   │   • OLS Linear Regression    │    │    • Responsive Themes      │  │
│   │   • Random Forest Regressor  │    │    • Subplots & Heatmap     │  │
│   │   • Automated Insights       │    │                             │  │
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

## Repository Structure

```
student-performance-dashboard/
│
├── README.md                      # Comprehensive academic & OST documentation
├── LICENSE                        # MIT license with explicit UCI dataset attribution
├── CODE_OF_CONDUCT.md             # Contributor Covenant v2.1 code of conduct
├── CONTRIBUTING.md                # Open-source collaboration & PR guide
├── SECURITY.md                    # Security policy & prohibited secrets protocol
├── CHANGELOG.md                   # Semantic versioning release history
├── .gitignore                     # Production Python & OS ignore patterns
├── .dockerignore                  # Docker build context optimization rules
├── requirements.txt               # Pinned Python package dependencies
├── Makefile                       # Unix / Git Bash task automation recipes
├── Dockerfile                     # Multi-stage secure non-root Docker build
├── docker-compose.yml             # Single-command container deployment
│
├── data/
│   ├── README.md                  # Dataset provenance, schema & ethical notes
│   └── raw/
│       ├── student-mat.csv        # Mathematics cohort data (395 records)
│       ├── student-por.csv        # Portuguese cohort data (649 records)
│       ├── student.txt            # Original attribute documentation
│       └── student-merge.R        # Cortez & Silva R script (382 overlapping students)
│
├── docs/
│   ├── dataset.md                 # Provenance, 33 attributes, and limitations
│   ├── methodology.md             # Engineering pipeline, stats & ML methodology
│   ├── architecture.md            # Decoupled system components & data flow
│   ├── git-workflow.md            # GitFlow model, Conventional Commits & PRs
│   └── linux-commands.md          # Open-source terminal commands reference
│
├── notebooks/
│   └── exploratory_analysis.ipynb # Jupyter exploration & validation notebook
│
├── src/
│   ├── __init__.py                # Package initialization & version metadata
│   ├── data_loader.py             # Data ingestion & schema boundary validation
│   ├── preprocessing.py           # Feature engineering & leakage-aware splits
│   ├── analysis.py                # Descriptive stats, correlations, ML & insights
│   └── visualizations.py          # 12+ publication-grade Plotly interactive figures
│
├── dashboard/
│   └── app.py                     # Interactive Streamlit multi-tab web application
│
├── tests/
│   ├── __init__.py                # Test package initialization
│   ├── test_data_loader.py        # Ingestion, schema and boundary tests (9 tests)
│   ├── test_preprocessing.py      # Label mapping, imputation & leakage tests (5 tests)
│   └── test_analysis.py           # Stats, correlation, ML & insight tests (7 tests)
│
├── assets/
│   └── screenshots/               # Application captures and visuals
│
└── .github/
    ├── workflows/
    │   └── tests.yml              # GitHub Actions CI matrix (Ubuntu/Windows, Py 3.10-3.12)
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md          # Structured bug report template
    │   └── feature_request.md     # Feature proposal template
    └── pull_request_template.md   # Standard pull request submission checklist
```

---

## Data Analysis

The analytical engine answers key questions regarding student achievement:

1. **Grade Distribution:** Grades range from 0 to 20. A noticeable subset of students (38 in Math, 9.6%) score exactly 0 due to exam absenteeism or school dropout. The median grade is 11.0, with an overall pass rate of 67.1% ($\ge 10/20$).
2. **Impact of Prior Failures:** Past academic failure is the single strongest negative demographic predictor. Students with 0 past failures average **11.26/20**, while students with 1 or more failures average **7.27/20**—a severe **3.99 point grade drop**.
3. **Weekly Study Time:** Studying $<2$ hours/week yields an average grade of **10.05/20**, whereas studying $\ge 5$ hours/week yields **11.41/20** ($+1.36$ point advantage).
4. **Parental Higher Education:** Students with at least one parent holding higher education achieve **11.76/20**, compared to **9.81/20** for students whose parents completed only primary school ($+1.95$ point difference).
5. **Absences:** School absences range from 0 to 93. Excessive absences ($>20$) correlate with elevated failure rates, although occasional absences among top students do not impair outcomes.

---

## Visualisations

The dashboard renders 12 interactive Plotly charts:
- **Final Grade ($G3$) Distribution:** Histogram with kernel density curve, mean marker, and passing threshold at 10.
- **Study Time vs Grade:** Box plot categorizing performance across weekly study ranges.
- **Absences vs Grade:** Scatter plot with least-squares OLS trendline.
- **Failures vs Grade:** Segmented box plot detailing performance drop across failure counts.
- **Gender Comparison:** Split violin and box plot showing comparative male vs female mark distributions.
- **School Comparison:** Gabriel Pereira (GP) vs Mousinho da Silveira (MS) box plot.
- **Age Distribution:** Stacked/grouped bar chart mapping age brackets against pass/fail status.
- **Parental Education:** Clustered bar chart comparing Mother's (`Medu`) and Father's (`Fedu`) educational levels against final score.
- **Internet Access:** Bar chart with standard deviation error bars illustrating home connectivity disparity.
- **Correlation Heatmap:** Interactive Pearson correlation matrix across all numeric features.
- **$G1/G2$ Collinearity Scatter:** Side-by-side scatter plots documenting the data leakage phenomenon.
- **Model Performance & Feature Importance:** Grouped bar charts comparing $R^2$ scores and top features.

---

## Machine Learning Methodology

Two regression algorithms are implemented:
1. **Ordinary Least Squares (OLS) Linear Regression**
2. **Random Forest Regressor (100 estimators, max depth 10, seed 42)**

### The Data Leakage Experiment
To demonstrate scientific integrity, models are evaluated under two distinct experimental setups:

- **Regime A (With $G1$ & $G2$ - Temporal Data Leakage):**
  Includes midterm exam scores. Because $G2$ has a Pearson correlation of $r = 0.9049$ with $G3$, models achieve an artificially inflated $R^2 \approx 0.816$. In reality, the algorithm is merely predicting that the final grade will equal the midterm grade.
- **Regime B (Without $G1$ & $G2$ - Early Warning Model):**
  Strictly excludes prior exam scores. Relies purely on demographic, social, and study habit indicators. Yields an honest baseline $R^2 \approx 0.268$. This model is actionable *before* examinations take place, providing true predictive utility.

---

## Results

*The following metrics were calculated on an unseen 20% test partition ($N=79$ test records) using the authentic Mathematics cohort ($N=395$):*

| Model Configuration | Regime | MAE | MSE | RMSE | $R^2$ Score | Features |
|---|---|---|---|---|---|---|
| **Random Forest Regressor** | **Regime A (With $G1/G2$ - Leakage)** | **1.1643** | **2.9199** | **1.7088** | **0.8161** | 41 |
| **Linear Regression** | **Regime A (With $G1/G2$ - Leakage)** | **1.6467** | **4.3871** | **2.0945** | **0.7241** | 41 |
| **Random Forest Regressor** | **Regime B (Without $G1/G2$ - Clean)** | **3.1064** | **11.6231** | **3.4093** | **0.2682** | 39 |
| **Linear Regression** | **Regime B (Without $G1/G2$ - Clean)** | **3.3953** | **13.6409** | **3.6934** | **0.1415** | 39 |

### Key Correlation Coefficients with Final Grade ($G3$):
- Second period grade ($G2$): **$+0.9049$**
- First period grade ($G1$): **$+0.8015$**
- Past class failures (`failures`): **$-0.3604$**
- Mother's education (`Medu`): **$+0.2171$**
- Student age (`age`): **$-0.1615$**
- Father's education (`Fedu`): **$+0.1525$**
- Workday alcohol consumption (`Dalc`): **$-0.0547$**
- Weekly study time (`studytime`): **$+0.0978$**

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Clone the Repository
```bash
git clone https://github.com/bhedheerbhushanjain-svg/student-performance-dashboard.git
cd student-performance-dashboard
```

### Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / Git Bash / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running Locally

To verify all components and test data ingestion from the command line:

```bash
python -c "from src.data_loader import load_raw_data; print(load_raw_data('mat').shape)"
```

---

## Running with Streamlit

Launch the interactive dashboard locally:

```bash
streamlit run dashboard/app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## Running with Docker

The repository includes a production-grade Docker container configuration.

### Using Docker Compose (Recommended)
```bash
docker compose up --build
```
Open `http://localhost:8501` in your browser.

To stop the container:
```bash
docker compose down
```

### Using Standalone Docker Build
```bash
# Build image
docker build -t student-performance-dashboard:latest .

# Run container
docker run -d --name student_dashboard -p 8501:8501 student-performance-dashboard:latest
```

---

## Git Workflow

The project follows the **GitFlow** branching strategy:
- `main`: Production-ready release branch.
- `develop`: Integration branch.
- `feature/*`: Dedicated topic branches (`feature/data-pipeline`, `feature/analysis-dashboard`, `feature/testing-ci`, `feature/docker`).
- `docs/*`: Dedicated documentation branch (`docs/project-documentation`).

### Conventional Commit Standard
Commits adhere to the Conventional Commits specification:
```text
feat: add dataset loading and preprocessing pipeline
feat: add statistical analysis and machine learning pipeline
feat: create comprehensive interactive Streamlit dashboard
test: add comprehensive test suite for data pipeline and analysis
ci: add GitHub Actions workflow for automated testing
build: add Docker containerization and Makefile automation
docs: add comprehensive documentation, issue templates, and licensing
```

---

## Linux/Git Bash Commands

The project demonstrates essential open-source terminal commands documented in [`docs/linux-commands.md`](docs/linux-commands.md):
- Directory and filesystem management: `pwd`, `ls -la`, `cd`, `mkdir -p`, `touch`, `cat`, `head`, `tail`, `grep`.
- Python runtime: `python -m venv`, `pip install`, `pip list`.
- Version control: `git status`, `git add`, `git commit`, `git branch`, `git checkout`, `git switch`, `git merge --no-ff`, `git log --graph`, `git remote -v`, `git push`.
- Task automation: `make install`, `make test`, `make run`.

---

## Testing

Testing is implemented using **pytest**. The test suite verifies dataset ingestion, schema boundaries, missing value handling, feature encoding, statistical metrics, ML evaluation, and automated insights.

Run tests locally:
```bash
pytest -v
```

### Actual Test Execution Results:
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\archive\student-performance-dashboard
collected 21 items

tests/test_analysis.py::TestAnalysis::test_compute_summary_statistics PASSED [  4%]
tests/test_analysis.py::TestAnalysis::test_compute_full_numeric_summary PASSED [  9%]
tests/test_analysis.py::TestAnalysis::test_compute_correlations PASSED   [ 14%]
tests/test_analysis.py::TestAnalysis::test_compute_subgroup_analysis PASSED [ 19%]
tests/test_analysis.py::TestAnalysis::test_train_and_evaluate_models PASSED [ 23%]
tests/test_analysis.py::TestAnalysis::test_compare_leakage_regimes PASSED [ 28%]
tests/test_analysis.py::TestAnalysis::test_generate_automated_insights PASSED [ 33%]
tests/test_data_loader.py::TestDataLoader::test_load_math_dataset_shape_and_columns PASSED [ 38%]
tests/test_data_loader.py::TestDataLoader::test_load_portuguese_dataset_shape_and_columns PASSED [ 42%]
tests/test_data_loader.py::TestDataLoader::test_load_both_courses_concatenation PASSED [ 47%]
tests/test_data_loader.py::TestDataLoader::test_load_merged_cohort_count PASSED [ 52%]
tests/test_data_loader.py::TestDataLoader::test_invalid_course_raises_value_error PASSED [ 57%]
tests/test_data_loader.py::TestDataLoader::test_missing_file_raises_file_not_found PASSED [ 61%]
tests/test_data_loader.py::TestDataLoader::test_validate_dataset_schema_constraints PASSED [ 66%]
tests/test_data_loader.py::TestDataLoader::test_validate_dataset_out_of_bounds_grade PASSED [ 71%]
tests/test_data_loader.py::TestDataLoader::test_validate_dataset_missing_column PASSED [ 76%]
tests/test_preprocessing.py::TestPreprocessing::test_add_readable_labels PASSED [ 80%]
tests/test_preprocessing.py::TestPreprocessing::test_handle_missing_values_clean_data PASSED [ 85%]
tests/test_preprocessing.py::TestPreprocessing::test_handle_missing_values_with_injected_nans PASSED [ 90%]
tests/test_preprocessing.py::TestPreprocessing::test_prepare_ml_features_without_prior_grades PASSED [ 95%]
tests/test_preprocessing.py::TestPreprocessing::test_prepare_ml_features_with_prior_grades PASSED [100%]

============================= 21 passed in 4.60s ==============================
```

---

## GitHub Actions

Automated CI is configured via [`.github/workflows/tests.yml`](.github/workflows/tests.yml).
- **Triggers:** On every `push` and `pull_request` to `main`, `develop`, and `feature/**` branches.
- **Matrix:** Executes across **Ubuntu** and **Windows** operating systems on **Python 3.10, 3.11, and 3.12**.
- **Steps:** Dependency caching, pip installation, package import verification, and verbose Pytest execution.

---

## Limitations

1. **Geographic Specificity:** Data reflects two Portuguese secondary schools in 2005–2006. Educational systems, grading policies, and cultural norms differ globally.
2. **Sample Size:** With 395 Mathematics and 649 Portuguese records, deep neural network modeling is unfeasible; tree ensembles and linear regressions are appropriate.
3. **Absence Anomaly:** Absences show weak linear correlation with final marks because both high-performing students who skip easy classes and disengaged students exhibit absences.
4. **Data Leakage in Academic Literature:** Many published studies report misleading $>80\%$ accuracies by incorporating $G1$ and $G2$. As shown in our results, true pre-semester prediction achieves $R^2 \approx 0.27$.

---

## Ethical/Data Considerations

- **Anonymity:** No personal identifying information (PII) such as student names, national IDs, addresses, or phone numbers are present.
- **Non-Causal Nature:** Statistical correlation does not establish causality. For example, high parental education correlates with student performance due to socio-economic opportunities, not innate intelligence.
- **Automated Decision-Making Warning:** Predictive models should serve as early-warning support tools for academic counseling and qualitative intervention—never as punitive criteria or automated streaming decisions.

---

## License

This project's original source code, scripts, dashboard implementation, and documentation are licensed under the **[MIT License](LICENSE)**.

The underlying **Student Performance Dataset** is attributed to Paulo Cortez and Alice Silva (University of Minho) and distributed under Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## Contributing

Contributions are welcomed! Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch naming conventions, Conventional Commit standards, and pull request procedures.

---

## Code of Conduct

All contributors and maintainers are committed to fostering a welcoming and harassment-free academic community. See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

---

## Author

- **Name:** Bhedheer Bhushan Jain
- **PRN:** `25030422033`
- **GitHub:** [@bhedheerbhushanjain-svg](https://github.com/bhedheerbhushanjain-svg)
- **Course:** Open Source Technologies (OST)
- **Institution:** Academic Project Submission
