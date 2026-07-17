@echo off
echo ==========================================
echo   AI English Conversation Tutor - Setup
echo ==========================================
echo.

echo Installing dependencies...
pip install -r requirements.txt
echo.

echo Starting server...
python server.py
pause
