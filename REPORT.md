# Project Report: Student Performance Analysis Dashboard
## An Academic Open Source Technologies (OST) Project

---

### Project Metadata & Student Credentials
- **Project Title:** Student Performance Analysis Dashboard
- **Course Title:** Open Source Technologies (OST)
- **Student Name:** Bhedheer Bhushan Jain
- **Permanent Registration Number (PRN):** `25030422033`
- **GitHub Account:** [`bhedheerbhushanjain-svg`](https://github.com/bhedheerbhushanjain-svg)
- **GitHub Repository:** [`student-performance-dashboard`](https://github.com/bhedheerbhushanjain-svg/student-performance-dashboard)
- **License:** MIT Open Source License
- **Academic Year:** 2026-2027

---

## Executive Summary / Abstract

The **Student Performance Analysis Dashboard** is a comprehensive, production-grade academic project engineered to demonstrate the end-to-end software development lifecycle within the Open Source Technologies (OST) paradigm. Utilizing empirical data from the University of California, Irvine (UCI) Machine Learning Repository—specifically the landmark educational dataset collected by Paulo Cortez and Alice Silva (2008)—this project examines socio-demographic, behavioral, and academic determinants of secondary student achievement in Mathematics and Portuguese language cohorts.

Beyond exploratory analytics, the project addresses a critical methodological challenge in educational data mining: **temporal data leakage**. Standard machine learning approaches often naively include mid-term grades ($G1$ and $G2$) to predict final grades ($G3$), producing spuriously high accuracy metrics ($R^2 > 0.81$) that fail to function as genuine early-warning diagnostic tools. This project formalizes and empirically evaluates two distinct modeling regimes:
1. **Regime A (Leaky Mid-Term Model):** Demonstrates the artificial correlation induced by mid-year exam scores.
2. **Regime B (Clean Early-Warning Model):** Restricts predictive features strictly to pre-enrollment variables (study habits, past failures, parental education, and home environment), yielding an honest, actionable early-intervention model.

The complete software system is constructed using modern open-source tooling, featuring a modular Python architecture (`src/`), an interactive Streamlit web dashboard with 7 thematic views (including an integrated Presentation view), 21 automated `pytest` test suites, containerization via Docker and Docker Compose, and continuous integration (CI) managed via multi-platform GitHub Actions workflows.

---

## 1. Introduction & Problem Statement

### 1.1 Background & Motivation
In secondary education, identifying students at risk of academic failure before semester examinations is crucial for timely institutional pedagogical intervention. Traditional institutional assessments often record failures after the academic term has concluded, precluding proactive mentoring or remedial instruction. 

Simultaneously, in the discipline of Open Source Technologies, students are required to master not only coding, but also the collaborative protocols, version control workflows, automated testing practices, legal compliance mechanisms, and documentation standards that govern contemporary open-source software development.

### 1.2 Problem Statement
Developing an educational analytics tool requires overcoming three interdisciplinary challenges:
1. **Data Ingestion & Integrity:** Educational survey datasets frequently feature mixed data types (binary, nominal, ordinal, and discrete numerical features) and disparate cohorts that require precise schema validation and deterministic joining.
2. **The Data Leakage Fallacy:** In typical student performance modeling, predictive pipelines naively include period 1 ($G1$) and period 2 ($G2$) evaluation marks to predict final period 3 ($G3$) performance. Because $G2$ and $G3$ exhibit Pearson correlation coefficients exceeding $0.90$, such models become simple identity mappings rather than predictive warning systems.
3. **Open Source Engineering Rigor:** Academic software projects often suffer from reproducibility failures, lacking container definitions, continuous integration testing, structured Git branching, and standardized community documentation.

### 1.3 Project Objectives
The primary objectives realized by this project are:
- **Architectural Modularity:** Decouple data loading, preprocessing, numerical analysis, visualization rendering, and web presentation into discrete, testable Python packages.
- **Empirical Rigor:** Ingest and analyze all 1,044 student records (395 Mathematics, 649 Portuguese, and 382 matched students) from the official UCI repository.
- **Data Leakage Mitigation:** Quantitatively compare and visualize the performance differential between leaky (Regime A) and early-warning (Regime B) machine learning pipelines using Random Forest and OLS Linear Regression.
- **Interactive Visualization:** Provide an accessible, responsive dashboard built with Streamlit and Plotly, incorporating real-time parametric filters and dynamic KPI cards.
- **Open Source Best Practices:** Enforce Git version control with a multi-branch git-flow model, automated CI via GitHub Actions across Ubuntu and Windows, reproducible Docker containers, and complete open-source legal documentation (MIT License, Code of Conduct, Contributing Guidelines, Security Policy).

---

## 2. Open Source Ecosystem & Technical Architecture

### 2.1 Technology Stack

| Layer | Technologies Selected | Rationale & Open-Source Role |
| :--- | :--- | :--- |
| **Core Language** | Python 3.11+ | High-performance scripting, universal scientific library support, and mature typing. |
| **Data Manipulation** | Pandas, NumPy | Vectorized DataFrame operations, schema validation, and numerical array transformations. |
| **Machine Learning** | Scikit-Learn | Standard library for OLS Regression, Random Forest Regressors, train/test splitting, and evaluation metrics ($R^2$, RMSE, MAE). |
| **Visual Analytics** | Plotly Express / Graph Objects | Interactive, client-side rendered charts (histograms, box plots, heatmaps, scatter plots). |
| **Presentation Tier** | Streamlit | Rapid, stateful, reactive web application framework supporting interactive sidebar filtering. |
| **Testing & QA** | Pytest | Industry-standard test runner with automated fixture discovery and assertion rewriting. |
| **Automation & CI** | GitHub Actions | Automated continuous integration executing test matrices across Ubuntu and Windows. |
| **Containerization** | Docker, Docker Compose | Immutable, non-root application image configuration ensuring zero environment drift. |
| **Task Automation** | GNU Make / Makefile | Standardized developer CLI interface (`make test`, `make run`, `make docker-build`). |
| **Version Control** | Git & GitHub | Distributed version control with semantic branching and pull request audit trails. |

---

### 2.2 System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                        WEB PRESENTATION LAYER                          │
│                                                                        │
│   Streamlit Web Interface (dashboard/app.py)                           │
│   ├─ Sidebar Controls & Multi-Select Filters                           │
│   ├─ Dynamic Real-Time KPI Metric Cards                                │
│   ├─ 7 Interactive Thematic Tabs:                                      │
│   │   1. 🏠 Overview & Architecture                                    │
│   │   2. 📁 Dataset Explorer (CSV Exporter)                            │
│   │   3. 📈 Performance Visualizations (12 Plotly Visuals)             │
│   │   4. 🧮 Statistical Deep-Dive (Parametric & Non-Parametric)        │
│   │   5. 🔬 ML & Data Leakage Lab (Early-Warning Simulator)            │
│   │   6. 💡 Automated Insights (Mathematically Generated Findings)     │
│   │   7. 📊 Presentation View (12 Live Academic Slides)                │
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
│   │   • One-Hot Categorical Encoding (drop_first=True)              │  │
│   │   • Regime Partitioning (Regime A: Leaky / Regime B: Clean)     │  │
│   └────────────────────────────────▲────────────────────────────────┘  │
│                                    │ Loads Raw Records                 │
│                                    ▼                                   │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     Data Loader Pipeline                        │  │
│   │                     (src/data_loader.py)                        │  │
│   │   • Safe File Resolution (resolve_data_path)                    │  │
│   │   • Semicolon CSV Parser                                        │  │
│   │   • Schema & Boundary Validation                                │  │
│   │   • Multi-Subject & Merged Cohort Reassembly (382 Overlaps)    │  │
│   └─────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Ingests
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA STORAGE                             │
│   data/raw/student-mat.csv (395 records)                               │
│   data/raw/student-por.csv (649 records)                               │
│   data/raw/student-merge.R (Reference Join Logic)                      │
│   data/raw/student.txt     (Original Codebook & Metadata)              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset & Domain Analysis

### 3.1 Provenance and Acquisition
The dataset utilized is the **UCI Student Performance Data Set**, collected by Paulo Cortez and Alice Silva at the University of Minho, Portugal (2008). The records were gathered from two public secondary schools in the Alentejo region of Portugal:
- **Gabriel Pereira (GP)**
- **Mousinho da Silveira (MS)**

The raw data was curated from school report cards and student demographic questionnaires across two academic subjects:
- **Mathematics (`student-mat.csv`):** 395 student instances.
- **Portuguese Language (`student-por.csv`):** 649 student instances.
- **Combined Corpus:** 1,044 total evaluated student records.
- **Matched Cohort:** 382 unique students shared across both subjects, linked via 13 demographic identity keys.

### 3.2 Schema Definition & Feature Taxonomy
The dataset contains 33 attributes per subject record, categorized into five thematic domains:

1. **Student Demographics:** `school`, `sex`, `age` (15–22), `address` (Urban/Rural), `famsize` (Binary $\le 3$ or $> 3$), `Pstatus` (Cohabitation status of parents: Together/Apart).
2. **Family & Social Background:** `Medu` (Mother education level 0–4), `Fedu` (Father education level 0–4), `Mjob` (Mother occupation), `Fjob` (Father occupation), `reason` (Reason for school choice), `guardian` (Mother/Father/Other).
3. **Study Habits & Academic History:** `traveltime` (Travel time to school 1–4), `studytime` (Weekly study hours: 1: $<2$h, 2: 2–5h, 3: 5–10h, 4: $>10$h), `failures` (Number of past class failures: 0–3, where 3 represents $\ge 3$).
4. **Institutional & Personal Support:** `schoolsup` (Extra educational support), `famsup` (Family educational support), `paid` (Extra paid classes), `activities` (Extracurricular participation), `nursery` (Attended nursery school), `higher` (Desire for higher education), `internet` (Home internet access), `romantic` (Romantic relationship status).
5. **Lifestyle & Health Metrics:** `famrel` (Quality of family relationships 1–5), `freetime` (Free time after school 1–5), `goout` (Going out with friends 1–5), `Dalc` (Workday alcohol consumption 1–5), `Walc` (Weekend alcohol consumption 1–5), `health` (Current health status 1–5), `absences` (Number of school absences 0–93).
6. **Academic Performance (Target Variables):**
   - $G1$: First period grade (discrete 0–20).
   - $G2$: Second period grade (discrete 0–20).
   - $G3$: Final year grade (discrete 0–20, target variable where passing threshold is $G3 \ge 10$).

---

## 4. Implementation Details

### 4.1 Data Loader (`src/data_loader.py`)
The ingestion pipeline guarantees data integrity through defensive programming:
- **`resolve_data_path()`:** Dynamically resolves data file locations across runtime environments (local development, test harnesses, and Docker containers).
- **`validate_dataset()`:** Asserts column completeness against `EXPECTED_COLUMNS`, enforces non-null constraints, and validates numerical boundaries:
  $$\text{age} \in [15, 25], \quad G1, G2, G3 \in [0, 20], \quad \text{studytime} \in [1, 4]$$
- **`load_both_courses()`:** Concatenates Math and Portuguese records into a unified 1,044-row DataFrame with a `subject` indicator.
- **`load_merged_cohort()`:** Implements the Cortez & Silva merge algorithm across 13 core keys:
  $$\text{Keys} = \{\text{school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, nursery, internet}\}$$

### 4.2 Preprocessing & Feature Engineering (`src/preprocessing.py`)
- **Missing Value Handling:** Incorporates automated median imputation for continuous features and mode imputation for categorical attributes.
- **One-Hot Encoding:** Applied to categorical features with `drop_first=True` to eliminate multi-collinearity and avoid the dummy variable trap.
- **Regime Separation:**
  - `prepare_features(df, include_period_grades=True)` $\rightarrow$ Regime A feature matrix (includes $G1, G2$).
  - `prepare_features(df, include_period_grades=False)` $\rightarrow$ Regime B feature matrix (excludes $G1, G2$).

### 4.3 Analytical & Statistical Engine (`src/analysis.py`)
Computes both parametric and non-parametric statistical indicators:
- **Parametric:** Sample Mean ($\bar{x}$), Sample Standard Deviation ($s$).
- **Non-Parametric:** Median, Interquartile Range ($\text{IQR} = Q_3 - Q_1$), Minimum, Maximum.
- **Distributional Shape:** Sample Skewness ($g_1$) and Sample Kurtosis ($g_2$).
- **Pearson Linear Correlation:**
  $$r_{X, Y} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

### 4.4 Interactive Presentation Engine (`dashboard/app.py`)
The dashboard is structured into 7 distinct operational tabs:
1. **🏠 Overview & Architecture:** High-level project objectives, system diagram, provenance, and ethical guidelines.
2. **📁 Dataset Explorer:** Raw data table with column glossary, record counts, and dynamic CSV export.
3. **📈 Performance Visualizations:** 12 Plotly interactive figures covering distributions, study hours, absences, and demographic splits.
4. **🧮 Statistical Deep-Dive:** Comprehensive metric grid with background gradient mapping and downloadable summary CSV.
5. **🔬 ML & Data Leakage Lab:** Live comparison of Regime A vs. Regime B, feature importance bar charts, and an interactive student risk prediction form.
6. **💡 Automated Insights:** Algorithmic observations computed directly from the current active filter slice.
7. **📊 Presentation View:** An integrated 12-slide dark-themed slide deck that dynamically recalculates metrics and graphs from the currently filtered cohort.

---

## 5. Machine Learning & The Data Leakage Paradigm

### 5.1 The Methodological Distinction

```
┌────────────────────────────────────────────────────────────────────────┐
│                   EDUCATIONAL DATA LEAKAGE ANALYSIS                    │
├───────────────────────────────────┬────────────────────────────────────┤
│ REGIME A: Leaky Mid-Term Model    │ REGIME B: Clean Early-Warning Model│
├───────────────────────────────────┼────────────────────────────────────┤
│ • Features: Demographic + G1 + G2 │ • Features: Demographic ONLY       │
│ • G2 Correlation with G3: r = 0.91│ • Excludes G1 and G2 completely    │
│ • Random Forest R² ≈ 0.816        │ • Random Forest R² ≈ 0.268         │
│ • Linear Regression R² ≈ 0.724    │ • Linear Regression R² ≈ 0.142     │
│ • High mathematical fit           │ • True diagnostic capability       │
│ • Flawed early intervention       │ • Deployable before classes begin  │
└───────────────────────────────────┴────────────────────────────────────┘
```

### 5.2 Empirical Evaluation Metrics
Models were trained with an 80/20 train-test split (`random_state=42`) on the full 1,044 student cohort. The results clearly illustrate the leakage phenomenon:

| Metric | Regime A (Leaky) - OLS | Regime A (Leaky) - RF | Regime B (Clean) - OLS | Regime B (Clean) - RF |
| :--- | :---: | :---: | :---: | :---: |
| **Coefficient of Determination ($R^2$)** | **0.7241** | **0.8161** | **0.1415** | **0.2682** |
| **Root Mean Squared Error (RMSE)** | 2.0134 | 1.6432 | 3.5381 | 3.2678 |
| **Mean Absolute Error (MAE)** | 1.2589 | 0.9842 | 2.6841 | 2.4512 |

### 5.3 Diagnostic Value of Regime B
While Regime B produces a lower nominal $R^2$ ($0.2682$), it represents the **true prognostic capacity** of pre-academic factors. Feature importance analysis via Random Forest demonstrates that:
1. **Past Failures (`failures`):** Dominates early risk detection (relative importance $> 22\%$).
2. **Weekly Study Time (`studytime`):** Demonstrates a consistent positive gradient.
3. **Parental Education (`Medu` / `Fedu`):** Accounts for measurable variance in student academic persistence.
4. **School Absences (`absences`):** Operates as an acute behavioral indicator of disengagement.

---

## 6. Empirical Findings from the 1,044 Student Dataset

Data extracted directly from the benchmark corpus reveals key educational insights:

### 6.1 Academic Performance Distributions
- **Overall Mean Final Grade ($G3$):** $11.34 / 20$ (Standard Deviation: $3.86$).
- **Overall Passing Rate ($G3 \ge 10$):** $78.0\%$ (814 passing, 230 failing).
- **Subject Disparity:**
  - Portuguese cohort average: $11.91 / 20$ (Passing rate: $84.6\%$).
  - Mathematics cohort average: $10.42 / 20$ (Passing rate: $67.1\%$).
- **Total Dropouts ($G3 = 0$):** 53 students across both cohorts scored 0 on their final evaluation, indicating examination abandonment or acute attrition.

### 6.2 Socio-Demographic Impacts
- **Urban vs. Rural Disparity:** Urban students average $11.62 / 20$, whereas rural students average $10.60 / 20$ (an advantage of $+1.02$ grade points for urban environments).
- **Home Internet Access:** Students with home connectivity ($79.2\%$) average $11.55 / 20$, compared to $10.53 / 20$ for students without connectivity ($+1.02$ grade points).
- **Study Time Gradient:**
  - $< 2$ hours/week: Mean grade of $10.58$ ($30.4\%$ of cohort).
  - $2–5$ hours/week: Mean grade of $11.34$ ($48.2\%$ of cohort).
  - $5–10$ hours/week: Mean grade of $12.49$ ($15.5\%$ of cohort).
  - $> 10$ hours/week: Mean grade of $12.27$ ($5.9\%$ of cohort).
  - *Observation:* Increasing weekly study time from $<2$h to $5–10$h yields an average improvement of $+1.91$ grade points ($+18\%$).
- **Impact of Past Failures:**
  - 0 past failures: Mean grade of $12.05$.
  - 1 past failure: Mean grade of $8.43$ (A drop of $-3.62$ grade points, below passing threshold).
  - 2 past failures: Mean grade of $7.48$.
  - 3 past failures: Mean grade of $6.80$.

---

## 7. Open Source Engineering & Development Workflow

### 7.1 Git Branching Strategy
The project follows a structured Git-Flow methodology. All work was organized into discrete branches before merging into `main`:

```
* 823ba1d (HEAD -> main, origin/main) chore: merge remote repository initialization
* 1a6559f Initial commit
* fab5346 merge: integrate dashboard usability improvements into main
| * e09f393 (origin/feature/dashboard-improvements) feat: improve dashboard usability with statistical export
|/  
* 921ce70 merge: release v0.1.0 to main branch
| * 617c0b4 (origin/feature/docker) feat: add production Docker containerization
|/  
| * a14b9c2 (origin/feature/testing-ci) feat: implement 21 pytest suites and GitHub Actions CI
|/  
| * b35d109 (origin/feature/analysis-dashboard) feat: build multi-tab Streamlit interface
|/  
| * f21a884 (origin/feature/data-pipeline) feat: construct data loader and preprocessing routines
|/  
* 5c04b82 (origin/develop) chore: establish development integration baseline
```

#### Active Branches in Repository:
1. `main`: Stable production branch with verified release tags.
2. `develop`: Core integration branch for ongoing engineering.
3. `feature/data-pipeline`: Data ingestion and validation modules.
4. `feature/analysis-dashboard`: Core Streamlit UI and Plotly visualizations.
5. `feature/testing-ci`: Automated test suite and GitHub Actions workflow.
6. `feature/docker`: Dockerfile, docker-compose, and container configs.
7. `docs/project-documentation`: Architecture, methodology, and dataset docs.
8. `feature/dashboard-improvements`: Usability enhancements and presentation view.

---

### 7.2 Open Source Governance & Legal Compliance
The repository includes standard open-source documentation required for open governance:
- **`LICENSE`:** Standard MIT Open Source License, granting unrestricted rights to inspect, modify, and distribute the software while preserving author attribution.
- **`CODE_OF_CONDUCT.md`:** Adopts the Contributor Covenant (version 2.1) establishing inclusive community participation standards.
- **`CONTRIBUTING.md`:** Detailed guidelines on code formatting, branch naming conventions, bug reporting, and pull request procedures.
- **`SECURITY.md`:** Security vulnerability reporting procedures and supported version disclosure.
- **`CHANGELOG.md`:** Semantic versioned record of all project iterations following Keep a Changelog standards.
- **`.github/ISSUE_TEMPLATE/`:** Standardized markdown templates for Bug Reports and Feature Requests.
- **`.github/pull_request_template.md`:** Structured template enforcing test verification and lint compliance before merging.

---

## 8. Quality Assurance, Testing & CI/CD

### 8.1 Automated Test Suites (`pytest`)
The project incorporates 21 unit and integration tests across three modules:

```
tests/
├── test_data_loader.py       # 9 tests
├── test_preprocessing.py     # 5 tests
└── test_analysis.py          # 7 tests
```

#### Verification Run Results:
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.4.1
rootdir: D:\archive\student-performance-dashboard
collected 21 items

tests/test_data_loader.py::test_load_raw_data_math_shape PASSED         [  4%]
tests/test_data_loader.py::test_load_raw_data_por_shape PASSED          [  9%]
tests/test_data_loader.py::test_load_raw_data_invalid_course PASSED     [ 14%]
tests/test_data_loader.py::test_expected_columns_present PASSED         [ 19%]
tests/test_data_loader.py::test_load_both_courses PASSED                [ 23%]
tests/test_data_loader.py::test_load_merged_cohort PASSED               [ 28%]
tests/test_data_loader.py::test_validate_dataset_valid PASSED            [ 33%]
tests/test_data_loader.py::test_validate_dataset_missing_column PASSED  [ 38%]
tests/test_data_loader.py::test_validate_dataset_out_of_bounds PASSED   [ 42%]
tests/test_preprocessing.py::test_add_readable_labels PASSED            [ 47%]
tests/test_preprocessing.py::test_impute_missing_values PASSED          [ 52%]
tests/test_preprocessing.py::test_encode_categorical_features PASSED    [ 57%]
tests/test_preprocessing.py::test_prepare_features_leakage_regime PASSED [ 61%]
tests/test_preprocessing.py::test_prepare_features_clean_regime PASSED  [ 66%]
tests/test_analysis.py::test_compute_summary_statistics PASSED          [ 71%]
tests/test_analysis.py::test_compute_full_numeric_summary PASSED        [ 76%]
tests/test_analysis.py::test_compute_correlations PASSED                [ 80%]
tests/test_analysis.py::test_compute_subgroup_analysis PASSED          [ 85%]
tests/test_analysis.py::test_train_grade_regressor_ols PASSED           [ 90%]
tests/test_analysis.py::test_compare_leakage_regimes PASSED            [ 95%]
tests/test_analysis.py::test_generate_automated_insights PASSED        [100%]

============================= 21 passed in 4.82s ==============================
```

### 8.2 Continuous Integration Pipeline (`.github/workflows/tests.yml`)
The GitHub Actions workflow executes on every `push` and `pull_request` against `main` and `develop`. It evaluates across an operating system and runtime version matrix:
- **Operating Systems:** `ubuntu-latest`, `windows-latest`
- **Python Runtimes:** `3.10`, `3.11`, `3.12`
- **Steps:** Dependency caching, package installation, pytest execution, and artifact verification.
- **Workflow Status:** Verified passing on GitHub Actions.

---

## 9. Containerization & Deployment

### 9.1 Docker Architecture
To guarantee environmental consistency across differing operating systems, the application is packaged in a lightweight container:
- **Base Image:** `python:3.11-slim`
- **Security:** Non-root execution user (`dashboarduser`, UID 1000)
- **Port:** `8501` exposed for Streamlit
- **Healthcheck:** Automated probe against `http://localhost:8501/_stcore/health`

### 9.2 Container Orchestration (`docker-compose.yml`)
```yaml
version: '3.8'

services:
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: student-performance-dashboard
    ports:
      - "8501:8501"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 10. Execution & User Guide

### 10.1 Local Execution via Python
```powershell
# 1. Clone repository
git clone https://github.com/bhedheerbhushanjain-svg/student-performance-dashboard.git
cd student-performance-dashboard

# 2. Configure environment (Windows PowerShell)
$env:PATH = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")

# 3. Install dependencies
pip install -r requirements.txt

# 4. Execute test suite
python -m pytest tests/ -v

# 5. Launch interactive dashboard
streamlit run dashboard/app.py
```
*Application available in browser at `http://localhost:8501`*

### 10.2 Execution via Docker
```bash
# Build and run containerized application
docker-compose up --build -d

# Verify container health
docker ps
```

---

## 11. Conclusion & Future Scope

### 11.1 Conclusion
The **Student Performance Analysis Dashboard** successfully bridges educational data science and rigorous open-source software engineering. By exposing and rectifying the **temporal data leakage** trap in secondary academic prediction, the project delivers an honest, actionable early-warning diagnostic tool. Simultaneously, the repository exemplifies open-source best practices through modular architecture, automated test coverage, multi-platform CI/CD, and transparent community governance.

### 11.2 Future Enhancements
- **Longitudinal Cohort Tracking:** Integrate multi-year datasets to evaluate whether remedial interventions successfully shift student trajectories across academic sessions.
- **Algorithmic Fairness Audits:** Implement fairness constraint toolkits (such as Fairlearn) to ensure predictive scores do not perpetuate historical socio-economic biases.
- **RESTful Inference API:** Expose a FastAPI microservice allowing third-party Learning Management Systems (LMS) like Moodle or Canvas to query student risk scores programmatically.

---

## 12. References

1. **Cortez, P., & Silva, A.** (2008). *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito & J. Teixeira (Eds.), Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal. EUROSIS-ETI, ISBN 978-9077381-39-7.
2. **UCI Machine Learning Repository.** (2014). *Student Performance Data Set*. Archived by University of California, Irvine, School of Information and Computer Sciences. `https://archive.ics.uci.edu/dataset/320/student+performance`
3. **Open Source Initiative (OSI).** (2007). *The MIT License (MIT)*. `https://opensource.org/licenses/MIT`
4. **Contributor Covenant.** (2021). *A Code of Conduct for Open Source Projects*, Version 2.1. `https://www.contributor-covenant.org/version/2/1/code_of_conduct/`
5. **Streamlit Inc.** (2024). *Streamlit Documentation: State, Caching, and Reactive Architecture*. `https://docs.streamlit.io`
6. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.

---
*Report compiled for the Open Source Technologies (OST) Academic Submission by Bhedheer Bhushan Jain (PRN: 25030422033).*
