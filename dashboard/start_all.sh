#!/bin/bash
# Start both backend and frontend

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================="
echo "Starting Heterogeneous Cluster Dashboard"
echo "========================================="

# Start backend in background
echo "Starting backend..."
cd "$SCRIPT_DIR/backend"
pip install -r requirements.txt > /dev/null 2>&1
python main.py &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
sleep 3

# Start frontend
echo "Starting frontend..."
cd "$SCRIPT_DIR/frontend"
npm install > /dev/null 2>&1

echo ""
echo "========================================="
echo "Dashboard is starting!"
echo "========================================="
echo "Backend API: http://localhost:8000"
echo "API Docs:    http://localhost:8000/docs"
echo "Frontend:    http://localhost:3000"
echo "========================================="
echo ""

npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
