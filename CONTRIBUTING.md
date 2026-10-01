# Contributing to Student Performance Analysis Dashboard

Thank you for your interest in contributing to the **Student Performance Analysis Dashboard**!
This is an academic Open Source Technologies (OST) project adhering to industry-standard version control and collaborative workflows.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Forking and Cloning](#forking-and-cloning)
   - [Environment Setup](#environment-setup)
3. [Git Workflow & Branching Strategy](#git-workflow--branching-strategy)
   - [Branch Naming Conventions](#branch-naming-conventions)
   - [Commit Message Guidelines](#commit-message-guidelines)
4. [Running Tests Locally](#running-tests-locally)
5. [Submitting a Pull Request](#submitting-a-pull-request)
6. [Data Science & Dataset Guidelines](#data-science--dataset-guidelines)
7. [Reporting Bugs and Requesting Features](#reporting-bugs-and-requesting-features)

---

## Code of Conduct

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat fellow contributors with respect and professionalism.

---

## Getting Started

### Prerequisites

- **Python:** Version 3.10, 3.11, or 3.12
- **Git:** Version 2.30+
- **Docker & Docker Compose:** (Optional, for containerized execution)

### Forking and Cloning

1. Fork the repository on GitHub:
   Navigate to `https://github.com/bhedheerbhushanjain-svg/student-performance-dashboard` and click the **Fork** button.

2. Clone your fork locally using Git Bash or terminal:

   ```bash
   git clone https://github.com/<your-username>/student-performance-dashboard.git
   cd student-performance-dashboard
   ```

3. Configure upstream remote to stay synchronized:

   ```bash
   git remote add upstream https://github.com/bhedheerbhushanjain-svg/student-performance-dashboard.git
   git remote -v
   ```

### Environment Setup

1. Create and activate a Python virtual environment:

   ```bash
   # On Linux / macOS / Git Bash
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Git Workflow & Branching Strategy

We follow the standard **GitFlow** branching model:

- `main`: Production-ready, stable releases.
- `develop`: Ongoing integration branch.
- Feature / Bugfix branches are branched off `develop` and merged back into `develop` via Pull Requests.

### Branch Naming Conventions

Always create a dedicated topic branch with a descriptive prefix:

* `feature/<short-description>`: New functionality or dashboard visual
* `fix/<bug-description>`: Bug fixes or data handling corrections
* `docs/<topic>`: Documentation updates or additions
* `test/<scope>`: Additional test cases or fixtures
* `refactor/<scope>`: Code refactoring without behavioral alterations

Example:

```bash
git checkout develop
git pull upstream develop
git checkout -b feature/add-attendance-analysis
```

### Commit Message Guidelines

We enforce the **Conventional Commits** specification:

```
<type>(<scope>): <short descriptive summary in imperative mood>

[optional body explaining rationale and context]
```

Standard types:
- `feat`: A new user-facing feature or visualization
- `fix`: A bug fix
- `docs`: Documentation changes only
- `test`: Adding or updating tests
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `build`: Changes to build system, Docker, or external dependencies
- `ci`: Changes to CI configuration files and scripts (e.g., GitHub Actions)
- `chore`: Maintenance tasks, `.gitignore` updates, etc.

Example:

```bash
git add src/visualizations.py
git commit -m "feat: add interactive absences distribution boxplot"
```

---

## Running Tests Locally

Before committing changes or opening a Pull Request, verify that all unit and integration tests pass:

```bash
# Run pytest with detailed verbose output
pytest -v

# Run pytest on a specific test module
pytest tests/test_data_loader.py -v
```

Ensure 100% test passing before pushing.

---

## Submitting a Pull Request

1. Push your topic branch to your GitHub fork:

   ```bash
   git push -u origin feature/add-attendance-analysis
   ```

2. Open a Pull Request on GitHub:
   - Base repository: `bhedheerbhushanjain-svg/student-performance-dashboard`
   - Base branch: `develop` (or `main` for hotfixes)
   - Compare branch: `your-username:feature/add-attendance-analysis`

3. Complete the [Pull Request Template](.github/pull_request_template.md) detailing:
   - Purpose and summary of changes
   - Related issue numbers (e.g., `Closes #3`)
   - Test results verification
   - Screenshots of dashboard UI modifications if applicable

4. Await peer review. Address any review comments using incremental commits on your topic branch.

---

## Data Science & Dataset Guidelines

1. **No Synthetic / Fake Data:** Never invent student records or manufacture artificial test sets. Use only the authentic UCI dataset in `data/raw/`.
2. **Privacy:** Do not commit any personal identifying information (PII).
3. **No Data Leakage Concealment:** Always maintain the strict separation between:
   - Regime A (With prior exam grades $G1$, $G2$): Documented as collinear / leaking.
   - Regime B (Without prior exam grades): Documented as genuine early-warning prediction.
4. **Honest Metrics:** Never fabricate model accuracies or evaluation figures ($R^2$, RMSE, MAE). All reported metrics must be mathematically calculated from the genuine dataset.

---

## Reporting Bugs and Requesting Features

Please use the issue templates provided in the repository:
- [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md)
- [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md)

Provide reproduction steps, expected behavior, system environment details, and screenshots where possible.
