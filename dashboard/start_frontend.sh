#!/bin/bash
# Start Next.js frontend

cd "$(dirname "$0")/frontend"

echo "Installing frontend dependencies..."
npm install

echo "Starting Next.js development server..."
echo "Dashboard at: http://localhost:3000"
npm run dev
