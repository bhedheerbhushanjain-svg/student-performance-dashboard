# Changelog

All notable changes to the **Student Performance Analysis Dashboard** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-01

### Added
- **Repository Setup & Governance:**
  - Standard open-source repository layout with `.gitignore` and `requirements.txt`.
  - Open source MIT License with explicit separation of original source code and UCI dataset attribution.
  - Contributor Covenant v2.1 Code of Conduct (`CODE_OF_CONDUCT.md`).
  - Step-by-step contribution guidelines (`CONTRIBUTING.md`).
  - Security policy preventing secret commits and PII exposure (`SECURITY.md`).
  - GitHub issue templates (`bug_report.md`, `feature_request.md`) and Pull Request template (`pull_request_template.md`).

- **Data Pipeline:**
  - Automated ingestion module (`src/data_loader.py`) supporting single-subject, concatenated, and linked-cohort datasets.
  - Conformance and schema validation with strict boundary checks on numeric grades and student ages.
  - Dataset preprocessing and readable label mapping decoders (`src/preprocessing.py`).
  - Missing-value imputation pipeline.
  - Multi-regime feature preparation separating Regime A (with G1/G2 leakage) from Regime B (clean early-warning features).

- **Statistical Analysis & Machine Learning:**
  - Descriptive statistics calculation module (`src/analysis.py`) including quartiles, IQR, skewness, and kurtosis.
  - Pearson correlation matrix computation with sorted target variable rankings.
  - Categorical subgroup aggregation.
  - Predictive modeling pipeline featuring Linear Regression and Random Forest Regressor.
  - Side-by-side data leakage comparative study (Regime A vs Regime B) measuring MAE, MSE, RMSE, and $R^2$.
  - Dynamic data-driven insights generator deriving empirical conclusions directly from active dataset distributions.

- **Visualizations & Streamlit Dashboard:**
  - Interactive Plotly visualization library (`src/visualizations.py`) with 12 publication-grade figures.
  - Full-featured multi-tab Streamlit web application (`dashboard/app.py`):
    - Real-time KPI summary cards (records, mean grade, median grade, passing rate, dropout counts).
    - Multi-criteria sidebar filtering (school, gender, age slider, study time, failures, internet).
    - Tab 1: Project Overview, Objectives, and System Architecture.
    - Tab 2: Interactive Dataset Explorer and CSV data export.
    - Tab 3: Exploratory Data Analysis with demographic and habit charts.
    - Tab 4: Statistical Deep-Dive and correlation analysis.
    - Tab 5: ML & Data Leakage Lab with live student grade predictor sandbox.
    - Tab 6: Dynamically computed automated analytical findings.

- **Testing & Continuous Integration:**
  - Comprehensive Pytest test suite (`tests/`) containing 21 unit and integration tests with 100% passing status.
  - GitHub Actions multi-OS (Ubuntu, Windows) and multi-version (Python 3.10, 3.11, 3.12) automated test workflow (`.github/workflows/tests.yml`).

- **Containerization & Build Automation:**
  - Production-ready `Dockerfile` based on `python:3.11-slim` with non-root security user and healthcheck.
  - `docker-compose.yml` for single-command deployment.
  - Linux / Git Bash automation `Makefile`.
  - Interactive Jupyter exploration notebook (`notebooks/exploratory_analysis.ipynb`).

- **Exhaustive Documentation:**
  - `docs/dataset.md`: Full provenance, attributes, and academic citation.
  - `docs/methodology.md`: Complete mathematical and engineering methodology.
  - `docs/architecture.md`: Component layout and data flow.
  - `docs/git-workflow.md`: Branching, commit conventions, and Pull Request workflow.
  - `docs/linux-commands.md`: Reference guide to open-source CLI tools.
  - Flagship `README.md` covering all academic evaluation requirements.
