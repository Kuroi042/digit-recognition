#!/usr/bin/env bash

set -e

VENV=".venv"

echo "==> Checking Python..."

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

echo "==> Creating virtual environment..."

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo "==> Activating virtual environment..."

source "$VENV/bin/activate"

echo "==> Installing dependencies..."

python3.12 -m pip install --upgrade pip
python3.12 -m pip install -r req.txt

echo "==> Starting GUI..."

python3.12 gui.py