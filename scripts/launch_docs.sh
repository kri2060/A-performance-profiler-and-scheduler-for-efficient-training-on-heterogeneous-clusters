#!/bin/bash
# Launch Documentation Website
# This script starts a web server to view the setup guides in your browser

set -e

echo "=================================================="
echo "📚 Multi-Node Setup Documentation Server"
echo "=================================================="
echo ""

# Get the directory where script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3 and try again"
    exit 1
fi

echo "✅ Python 3 found"

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Flask not found. Installing dependencies..."

    # Try to install in virtual environment first
    if [ ! -d "docs-website/venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv docs-website/venv
    fi

    echo "Activating virtual environment..."
    source docs-website/venv/bin/activate

    echo "Installing Flask and dependencies..."
    pip install -q -r docs-website/requirements.txt

    echo "✅ Dependencies installed"
else
    echo "✅ Flask already installed"
fi

# Get local IP address
if command -v hostname &> /dev/null; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
else
    LOCAL_IP="localhost"
fi

echo ""
echo "🚀 Starting documentation server..."
echo ""
echo "=================================================="
echo "📖 Access the documentation at:"
echo ""
echo "   Local:   http://localhost:5000"
if [ "$LOCAL_IP" != "localhost" ]; then
    echo "   Network: http://$LOCAL_IP:5000"
fi
echo ""
echo "=================================================="
echo ""
echo "Available guides:"
echo "  ⚡ Quick Start (5 Minutes)"
echo "  ✅ Setup Checklist"
echo "  🎯 Master Node Setup"
echo "  🔧 Worker Node Setup"
echo "  🚀 Launch Guide"
echo "  🎨 Visual Setup Diagrams"
echo "  🐳 Docker Multi-Node Guide"
echo ""
echo "=================================================="
echo "Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Activate venv if it exists
if [ -d "docs-website/venv" ]; then
    source docs-website/venv/bin/activate
fi

# Start the Flask app
cd docs-website
python3 app.py
