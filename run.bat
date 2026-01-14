@echo off
chcp 65001 >nul
echo ============================================================
echo WATERMARK REMOVER with LaMa Deep Learning
echo ============================================================
echo.

REM Check if venv311 exists
if exist "venv311\Scripts\python.exe" (
    echo [OK] Virtual environment found!
    goto :run_app
)

echo [!] Virtual environment not found. Setting up...
echo.

REM Check if Python 3.11 is available
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.11 is not installed!
    echo.
    echo Please install Python 3.11 from:
    echo https://www.python.org/downloads/release/python-3119/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment with Python 3.11...
py -3.11 -m venv venv311
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

echo [2/3] Upgrading pip...
venv311\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [3/3] Installing dependencies (this may take a few minutes)...
venv311\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo [OK] Setup completed successfully!
echo.

:run_app
echo Starting Watermark Remover...
echo.
venv311\Scripts\python.exe watermark_remover.py

pause
