# Linux & Open Source Terminal Commands Reference

This document catalogs the essential Linux, Git Bash, and open-source CLI tools utilized during the development, maintenance, and deployment of the **Student Performance Analysis Dashboard**.

---

## 1. Directory and File System Operations

### `pwd` (Print Working Directory)
Prints the current absolute directory path.
```bash
# Verify current working repository location
pwd
# Output: /d/archive/student-performance-dashboard
```

### `ls` (List Directory Contents)
Lists directory files and permissions.
```bash
# List all files including hidden dotfiles with human-readable sizes
ls -la
# List only the raw dataset directory
ls -lh data/raw/
```

### `cd` (Change Directory)
Navigates through the directory tree.
```bash
# Switch to the student performance project directory
cd /d/archive/student-performance-dashboard

# Navigate up one directory level
cd ..
```

### `mkdir` (Make Directory)
Creates new directories.
```bash
# Create nested project directory structures recursively
mkdir -p data/raw docs notebooks src dashboard tests assets/screenshots .github/workflows
```

### `touch` (Create Empty File / Update Timestamp)
Creates new empty files or updates access timestamps.
```bash
# Create package initialization files
touch src/__init__.py tests/__init__.py
```

### `cat` (Concatenate and Display File Content)
Displays file contents directly in terminal.
```bash
# Inspect the raw dataset merge script
cat data/raw/student-merge.R

# Concatenate requirements
cat requirements.txt
```

### `grep` (Search Text with Regular Expressions)
Filters file content or command output for matching patterns.
```bash
# Search for target column usage across source files
grep -rn "G3" src/

# Search for all test functions in tests directory
grep -rn "def test_" tests/
```

### `head` (View Beginning of File)
Outputs the first $N$ lines of a file (default 10).
```bash
# Inspect the header and first 5 student records in Portuguese course
head -n 6 data/raw/student-por.csv
```

### `tail` (View End of File)
Outputs the last $N$ lines of a file (default 10).
```bash
# Inspect the last records of Mathematics course
tail -n 5 data/raw/student-mat.csv
```

---

## 2. Python & Package Management Commands

### `python` (Python Interpreter)
Executes scripts, modules, and inline statements.
```bash
# Check installed Python version
python --version

# Verify module import and execute inline sanity check
python -c "import pandas, sklearn, streamlit; print('Environment ready!')"
```

### `pip` (Python Package Installer)
Manages third-party library dependencies.
```bash
# Upgrade pip package manager
python -m pip install --upgrade pip

# Install project dependencies from requirements.txt
pip install -r requirements.txt

# Inspect installed package versions
pip list | grep -E "pandas|streamlit|scikit-learn|pytest"
```

---

## 3. Git Version Control Commands

### `git status`
Displays the state of the working directory and staging area.
```bash
git status
```

### `git add`
Stages file changes for the next commit.
```bash
# Stage specific files
git add src/data_loader.py

# Stage all tracked and untracked modifications
git add .
```

### `git commit`
Records staged snapshots into project history.
```bash
# Commit with conventional commit message
git commit -m "feat: add dataset loading and preprocessing pipeline"
```

### `git branch`
Lists, creates, or deletes branches.
```bash
# List all local branches with current branch indicated by an asterisk
git branch

# List both local and remote branches
git branch -a
```

### `git checkout` & `git switch`
Switches branches or restores working tree files.
```bash
# Create and switch to a new feature branch
git checkout -b feature/data-pipeline

# Modern syntax to switch branches
git switch develop
```

### `git merge`
Combines history from one branch into the current branch.
```bash
# Merge feature branch with an explicit non-fast-forward merge commit
git checkout develop
git merge --no-ff feature/data-pipeline -m "merge: integrate data pipeline into develop"
```

### `git log`
Displays chronological commit history.
```bash
# Compact one-line graph view of branches and merges
git log --oneline --graph --decorate --all

# View the last 5 commits with full author and diff summary
git log -n 5 --stat
```

### `git remote`
Manages connections to remote repositories.
```bash
# Display configured remotes with push/fetch URLs
git remote -v

# Add origin remote URL
git remote add origin git@github.com:bhedheerbhushanjain-svg/student-performance-dashboard.git
```

### `git push`
Uploads local branch commits to the remote repository.
```bash
# Push main branch and set upstream tracking
git push -u origin main

# Push develop branch
git push origin develop
```

### `git pull`
Fetches and merges commits from remote upstream branch into local branch.
```bash
git pull origin develop
```

---

## 4. Container & Build Automation Commands

### `make` (Makefile Automation)
Runs automated recipes defined in `Makefile`.
```bash
# Execute test suite
make test

# Launch local dashboard
make run
```

### `docker` & `docker compose`
Builds and manages isolated application containers.
```bash
# Build the application container image
docker build -t student-performance-dashboard:latest .

# Run container with port forwarding to 8501
docker run -p 8501:8501 student-performance-dashboard:latest

# Build and start container stack using Compose
docker compose up --build -d

# Stop running container stack
docker compose down
```
