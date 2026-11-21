#!/usr/bin/env bash

set -e

PYTHON="python3.13"
VENV=".venv-py313"

echo "==> Checking Python..."

if ! command -v "$PYTHON" &> /dev/null; then
    echo "Error: Python 3.12 is required but is not installed."
    exit 1
fi

echo "==> Creating virtual environment..."

if [ ! -d "$VENV" ]; then
    "$PYTHON" -m venv "$VENV"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo "==> Activating virtual environment..."

source "$VENV/bin/activate"

echo "==> Installing dependencies..."

python -m pip install --upgrade pip
python -m pip install -r req.txt

echo "==> Starting GUI..."

python gui.py
