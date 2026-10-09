@echo off
title CropMLQ-Sense
cd /d "%~dp0"

echo Starting CropMLQ-Sense...
echo.

start "" /min cmd /c "python -m streamlit run app.py"

timeout /t 5 /nobreak >nul

start "" "http://localhost:8501"

echo.
echo CropMLQ-Sense is opening in your browser.
echo Keep this window open while using the website.
echo.
pause
