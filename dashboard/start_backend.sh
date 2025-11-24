#!/bin/bash
# Start FastAPI backend

cd "$(dirname "$0")/backend"

echo "Installing backend dependencies..."
pip install -r requirements.txt

echo "Starting FastAPI server on http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
python main.py
