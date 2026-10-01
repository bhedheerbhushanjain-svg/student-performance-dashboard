# Student Performance Analysis Dashboard - Automation Makefile
# Demonstrates Linux / Unix command-line build automation for OST projects

.PHONY: help install test run clean docker-build docker-run docker-up docker-down

PYTHON ?= python
PIP ?= pip
STREAMLIT ?= streamlit
PYTEST ?= pytest

help:
	@echo "Student Performance Analysis Dashboard - Available Commands:"
	@echo "  make install       Install all Python package dependencies"
	@echo "  make test          Run pytest test suite with verbosity"
	@echo "  make run           Launch Streamlit web dashboard locally"
	@echo "  make clean         Remove Python byte-cache and pytest artifacts"
	@echo "  make docker-build  Build Docker container image"
	@echo "  make docker-run    Run standalone Docker container"
	@echo "  make docker-up     Launch container stack using docker-compose"
	@echo "  make docker-down   Stop and remove running docker containers"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -v

run:
	$(STREAMLIT) run dashboard/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t student-performance-dashboard:latest .

docker-run:
	docker run -d --name student_dashboard -p 8501:8501 student-performance-dashboard:latest

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
