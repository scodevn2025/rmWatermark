@echo off
echo ============================================================
echo WATERMARK REMOVER with LaMa Deep Learning
echo ============================================================
echo.

REM Use Python 3.11 virtual environment for LaMa support
if exist "venv311\Scripts\python.exe" (
    echo Using Python 3.11 with LaMa model...
    venv311\Scripts\python.exe watermark_remover.py
) else (
    echo WARNING: venv311 not found, using system Python
    echo For best quality, run: py -3.11 -m venv venv311
    python watermark_remover.py
)

pause
