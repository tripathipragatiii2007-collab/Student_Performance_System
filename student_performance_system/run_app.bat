@echo off
echo ========================================================
echo   Starting Student Performance Report System...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
