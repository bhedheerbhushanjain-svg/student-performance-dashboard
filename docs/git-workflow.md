# Git Workflow & Open Source Collaboration Guide

## Git Workflow Strategy: GitFlow

This project demonstrates the **GitFlow** branching model, a standard branching strategy used across mature open-source software projects and enterprise engineering teams.

---

## 1. Branch Topology

```
main        ───●─────────────────────────────────────────────────────────────●── (v0.1.0 Release)
                \                                                           /
develop          ───●───────────●───────────────●───────────────●──────────●──── (Integration)
                     \         / \             / \             / \        /
feature/data-pipe     ───●───●──  \           /   \           /   \      /
                                   ───●───●──      \         /     \    /
feature/analysis-dash                               ───●───●──      \  /
                                                                     ──● (docs & ci)
```

### Branch Roles
1. **`main`**: The canonical production branch. Contains strictly verified, release-tagged, stable code.
2. **`develop`**: The primary integration branch where features are aggregated prior to tagging a release.
3. **`feature/*`**: Topic branches created for discrete enhancements (e.g. `feature/data-pipeline`, `feature/analysis-dashboard`, `feature/docker`).
4. **`docs/*`**: Topic branches dedicated to documentation, licensing, and workflow guides.
5. **`fix/*`**: Dedicated branches for resolving defects identified during integration.

---

## 2. Step-by-Step Feature Workflow

### Step 1: Synchronize and Branch
Always branch from the latest state of `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/interactive-predictor
```

### Step 2: Implement Changes and Test
Make modular, focused edits to the codebase. Verify tests pass locally:
```bash
pytest -v
```

### Step 3: Stage and Commit Following Conventional Commits
```bash
git status
git add src/analysis.py src/visualizations.py
git commit -m "feat: add confidence interval visualization for predictions"
```

### Step 4: Push to Remote Topic Branch
```bash
git push -u origin feature/interactive-predictor
```

### Step 5: Open a Pull Request
On GitHub, open a PR from `feature/interactive-predictor` into `develop`.
- Fill out the PR template.
- Link related issues (e.g., `Closes #4`).
- Verify automated GitHub Actions CI passes.

### Step 6: Review and Merge
Perform non-fast-forward merge (`--no-ff`) to preserve branch topology and history:
```bash
git checkout develop
git merge --no-ff feature/interactive-predictor -m "merge: integrate interactive predictor into develop"
```

---

## 3. Conventional Commit Standards

Every commit message follows the format:
```
<type>(<scope>): <subject>

[optional body]
```

### Commit Types Demonstrated in this Repository:
- `chore`: Project structure initialization, `.gitignore` setup.
- `feat`: Addition of data loader, preprocessing, analysis, and Streamlit dashboard.
- `test`: Unit and integration test suite implementation.
- `ci`: GitHub Actions test workflow setup.
- `build`: Dockerfile, docker-compose, and Makefile creation.
- `docs`: Documentation, README, and license authoring.
- `fix`: Bug corrections and safe missing value handling.
