@echo off
REM Launch Documentation Website (Windows)
REM This script starts a web server to view the setup guides in your browser

echo ==================================================
echo 📚 Multi-Node Setup Documentation Server
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed
    echo Please install Python 3 and try again
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo 📦 Flask not found. Installing dependencies...

    REM Create virtual environment if it doesn't exist
    if not exist "docs-website\venv" (
        echo Creating virtual environment...
        python -m venv docs-website\venv
    )

    echo Activating virtual environment...
    call docs-website\venv\Scripts\activate.bat

    echo Installing Flask and dependencies...
    pip install -q -r docs-website\requirements.txt

    echo ✅ Dependencies installed
) else (
    echo ✅ Flask already installed
)

REM Get local IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set LOCAL_IP=%%a
    goto :found_ip
)
:found_ip
set LOCAL_IP=%LOCAL_IP:~1%

echo.
echo 🚀 Starting documentation server...
echo.
echo ==================================================
echo 📖 Access the documentation at:
echo.
echo    Local:   http://localhost:5000
if defined LOCAL_IP echo    Network: http://%LOCAL_IP%:5000
echo.
echo ==================================================
echo.
echo Available guides:
echo   ⚡ Quick Start (5 Minutes)
echo   ✅ Setup Checklist
echo   🎯 Master Node Setup
echo   🔧 Worker Node Setup
echo   🚀 Launch Guide
echo   🎨 Visual Setup Diagrams
echo   🐳 Docker Multi-Node Guide
echo.
echo ==================================================
echo Press Ctrl+C to stop the server
echo ==================================================
echo.

REM Activate venv if it exists
if exist "docs-website\venv" (
    call docs-website\venv\Scripts\activate.bat
)

REM Start the Flask app
cd docs-website
python app.py
