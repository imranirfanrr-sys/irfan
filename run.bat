@echo off
echo Installing required packages...
py -m pip install -r requirements.txt
echo.
echo Starting the AI Diagnostics Server...
echo Please open your browser and go to http://127.0.0.1:5000/
py app.py
pause
