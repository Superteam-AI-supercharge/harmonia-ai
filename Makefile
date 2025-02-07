.PHONY: help install run run-bot venv clean setup dirs env-sample test lint

# Python variables
PYTHON := python3
VENV_NAME := newenv
VENV_BIN := $(VENV_NAME)/bin
VENV_ACTIVATE := . $(VENV_BIN)/activate

# Create and activate virtual environment
venv:
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Virtual environment created. Activate it with: source $(VENV_NAME)/bin/activate"

# Install all dependencies
install:
	pip install fastapi uvicorn python-telegram-bot python-dotenv langchain faiss-cpu \
		sentence-transformers PyPDF2 python-docx groq

# Create necessary directories
dirs:
	mkdir -p superteam_data/superteam
	mkdir -p superteam_data/solana
	mkdir -p superteam_data/superteam_vn
	mkdir -p superteam_data/superteam_nig
	mkdir -p superteam_data/bounties
	mkdir -p superteam_data/grants
	mkdir -p superteam_data/hackatons
	mkdir -p temp

# Create .env.sample from .env
env-sample:
	@if [ -f .env ]; then \
		grep -v '^#' .env | cut -d '=' -f1 | sed 's/$$/ =/' > .env.sample; \
		echo ".env.sample created with the following template:"; \
		echo "GROQ_API_KEY ="; \
		echo "TELEGRAM_BOT_TOKEN ="; \
	else \
		echo "Creating new .env.sample..."; \
		echo "GROQ_API_KEY =" > .env.sample; \
		echo "TELEGRAM_BOT_TOKEN =" >> .env.sample; \
	fi

# Run FastAPI server
run:
	uvicorn main:app --reload

# Run Telegram bot
run-bot:
	$(PYTHON) telegram_bot.py

# Full setup command
setup: venv install dirs env-sample
	@echo "Project setup complete. Don't forget to:"
	@echo "1. Activate your virtual environment: source $(VENV_NAME)/bin/activate"
	@echo "2. Fill in your .env file with the required API keys"

# Clean up temporary files and cache
clean:
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf temp/*

# Run tests (placeholder - add your test command)
test:
	$(PYTHON) -m pytest

# Run linter (placeholder - add your preferred linter)
lint:
	$(PYTHON) -m flake8 .

# Help command
help:
	@echo "Available commands:"
	@echo "  make setup      - Full project setup (venv, install, directories, env)"
	@echo "  make venv       - Create virtual environment"
	@echo "  make install    - Install Python dependencies"
	@echo "  make dirs       - Create project directory structure"
	@echo "  make env-sample - Create .env.sample template"
	@echo "  make run        - Run FastAPI server"
	@echo "  make run-bot    - Run Telegram bot"
	@echo "  make clean      - Clean temporary files and cache"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"